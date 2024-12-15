//===--- SiFive_RISCVExpandVXPseudo.cpp - Expand Pseudo VX instructions ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass scans for VX instructions in inner most loops, and try to hoist
// them out of the loop based a regional pressure analysis. This helps improve
// performance on some cores where scalar to vector is slow.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <numeric>
#include <vector>

using namespace llvm;

static cl::opt<bool> EnableExpandVXPseudoOpt(
    "riscv-enable-expand-vxpseudo", cl::init(true), cl::Hidden,
    cl::desc("Enable expand vx pseudos into vv pseudos"));

#define DEBUG_TYPE "riscv-expand-vxpseudo"
#define RISCV_EXPAND_VXPSEUDO_NAME "RISC-V Expand VXPseudo pass"

namespace llvm::RISCV {
struct RISCVVXPseudoInfo {
  uint16_t VXPseudo;
  uint16_t VVPseudo;
};

#define GET_RISCVVXPseudosTable_DECL
#define GET_RISCVVXPseudosTable_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace llvm::RISCV

typedef SmallVector<unsigned, 8> RegPressureSet;

namespace {

class RISCVExpandVXPseudo : public MachineFunctionPass {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *MLI = nullptr;
  MachineDominatorTree *DT = nullptr;
  // For calcMICost to keep track of seen defs
  SmallSet<Register, 32> RegSeen;
  // Recursion Limit for isHoitable()
  const int DepthLimit = 2;

public:
  static char ID;

  RISCVExpandVXPseudo() : MachineFunctionPass(ID) {
    initializeRISCVExpandVXPseudoPass(*PassRegistry::getPassRegistry());
    }
    bool runOnMachineFunction(MachineFunction &) override;
    StringRef getPassName() const override {
      return RISCV_EXPAND_VXPSEUDO_NAME; }
    int unfuseVX(const std::vector<MachineInstr *> &);
    unsigned getMVInst(RISCVII::VLMUL, const TargetRegisterClass *);
    RegPressureSet getLoopMaxPressure(const MachineLoop *);
    bool tryHoistSrcDef(const MachineLoop *, RegPressureSet &, MachineInstr *);
    bool isHoistable(const MachineLoop *, llvm::DenseMap<unsigned, int> &,
                     MachineInstr *, int);
    void recursiveHoist(const MachineLoop *, MachineInstr *);
    DenseMap<unsigned, int> calcMICost(const MachineInstr &);
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<MachineLoopInfoWrapperPass>();
      AU.addRequired<MachineDominatorTreeWrapperPass>();
      AU.addPreserved<MachineLoopInfoWrapperPass>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
};

} // end anonymous namespace

char RISCVExpandVXPseudo::ID = 0;
char &llvm::RISCVExpandVXPseudoID = RISCVExpandVXPseudo::ID;

INITIALIZE_PASS_BEGIN(RISCVExpandVXPseudo, DEBUG_TYPE, RISCV_EXPAND_VXPSEUDO_NAME,
                false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVExpandVXPseudo, DEBUG_TYPE, RISCV_EXPAND_VXPSEUDO_NAME,
                false, false)

/// Returns an instance of the Expand VXPseudo pass.
FunctionPass *llvm::createRISCVExpandVXPseudoPass() {
  return new RISCVExpandVXPseudo();
}

bool RISCVExpandVXPseudo::runOnMachineFunction(MachineFunction &MF) {

  const TargetSubtargetInfo &ST = MF.getSubtarget();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  LLVM_DEBUG(dbgs() << "******** ExpandVXPseudo: ");
  LLVM_DEBUG(dbgs() << MF.getName() << " ********\n");

  std::vector<MachineInstr *> VXInstrs;
  for (auto &MBB: MF) {
    /* Only process inner most loop */
    auto *Loop = MLI->getLoopFor(&MBB);
    if ((!Loop) || (!Loop->isInnermost()))
      continue;

    for (auto &MI: MBB) {
      const RISCV::RISCVVXPseudoInfo *I = RISCV::getVXPseudoInfo(MI.getOpcode());
      if (I && (I->VXPseudo != I->VVPseudo))
        VXInstrs.push_back(&MI);
    }
  }
  LLVM_DEBUG(dbgs() << "# Expand Candidates: " << VXInstrs.size() << "\n");

  return unfuseVX(VXInstrs) != 0;
}

// This is basically modified from early machine LICM
RegPressureSet
RISCVExpandVXPseudo::getLoopMaxPressure(const MachineLoop *Loop) {
  // Perform a DFS walk to determine the order of visit.
  SmallVector<MachineDomTreeNode *, 8> WorkList;

  RegPressureSet MaxPressure = RegPressureSet(TRI->getNumRegPressureSets(), 0);
  RegPressureSet Pressure = RegPressureSet(MaxPressure.size(), 0);
  RegSeen.clear();

  // Need to calculate the pressure of preheader first
  auto *Preheader = Loop->getLoopPreheader();
  for (auto &MI : *Preheader)
    for (auto &[Class, Cost] : calcMICost(MI))
      Pressure[Class] += Cost;

  auto *HeaderN = DT->getNode(Loop->getHeader());
  WorkList.push_back(HeaderN);

  while (!WorkList.empty()) {
    MachineDomTreeNode *Node = WorkList.pop_back_val();
    assert(Node && "Null dominator tree node?");

    MachineBasicBlock *BB = Node->getBlock();
    for (auto &MI : *BB) {
      for (auto &[Class, Cost] : calcMICost(MI)) {
        Pressure[Class] += Cost;
        MaxPressure[Class] = std::max(MaxPressure[Class], Pressure[Class]);
      }
    }

    // If the header of the loop containing this basic block is a landing pad,
    // then don't try to hoist instructions out of this loop.
    if (Loop->getHeader()->isEHPad())
      continue;

    unsigned NumChildren = Node->getNumChildren();

    // Don't consider a large switch statement
    if (BB->succ_size() >= 25)
      NumChildren = 0;

    if (NumChildren) {
      // Add children in reverse order as then the next popped worklist node is
      // the first child of this node.  This means we ultimately traverse the
      // DOM tree in exactly the same order as if we'd recursed.
      for (MachineDomTreeNode *Child : reverse(Node->children()))
        WorkList.push_back(Child);
    }
  }
  return MaxPressure;
}

DenseMap<unsigned, int>
RISCVExpandVXPseudo::calcMICost(const MachineInstr &MI) {

  DenseMap<unsigned, int> Cost;
  for (auto &MO : MI.explicit_operands()) {
    if (!MO.isReg())
      continue;

    Register Reg = MO.getReg();
    if (!Reg.isVirtual())
      continue;

    int RegAccFactor;
    if (MO.isKill()) {
      RegSeen.erase(Reg);
      RegAccFactor = -1;
    } else if (MO.isDef()) {
      if (RegSeen.contains(Reg))
        continue;
      RegSeen.insert(Reg);
      RegAccFactor = 1;
    } else {
      continue;
    }

    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    const RegClassWeight W = TRI->getRegClassWeight(RC);

    const int *PS = TRI->getRegClassPressureSets(RC);
    for (; *PS != -1; ++PS)
      Cost[*PS] += W.RegWeight * RegAccFactor;
  }
  return Cost;
}

int RISCVExpandVXPseudo::unfuseVX(const std::vector<MachineInstr *> &VXInstrs) {
  int HoistedCount = 0;
  for (auto &MI : VXInstrs) {
    Register DstReg = MI->getOperand(0).getReg();
    LLVM_DEBUG(dbgs() << "Dst: "; MI->getOperand(0).dump());
    const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);

    auto *Loop = MLI->getLoopFor(MI->getParent());
    assert(Loop && "Null machine loop?");
    if (Loop->getLoopPreheader() == nullptr)
      continue;

    // Find the scalar source reg
    Register ScalarReg = -1;
    int ScalarRegIdx = -1;
    for (auto &MO : MI->uses()) {
      if (!MO.isReg())
        continue;

      ScalarReg = MO.getReg();
      if (!ScalarReg.isVirtual())
        continue;
      auto *ScalarRC = MRI->getRegClass(ScalarReg);
      if ((ScalarRC == &RISCV::GPRRegClass) ||
          (ScalarRC == &RISCV::FPR16RegClass) ||
          (ScalarRC == &RISCV::FPR32RegClass) ||
          (ScalarRC == &RISCV::FPR64RegClass)) {
        ScalarRegIdx = MO.getOperandNo();
        break;
      }
    }
    if (ScalarRegIdx == -1)
      continue;

    LLVM_DEBUG(dbgs() << "******* Loop ********\n");

    // Expand only if vreg pressure of the inner most loop is under the limit
    auto MaxPressure = getLoopMaxPressure(Loop);
    auto PressureLimit = TRI->getRegPressureSetLimit(*MI->getMF(), RISCV::VM);
    if (MaxPressure[RISCV::VM] >= PressureLimit)
      continue;

    // If the source of the splat is definied within the same loop,
    // try to hoist the def too if the pressure allow us to do so
    auto *ScalarDef = MRI->getVRegDef(ScalarReg);
    if (Loop->contains(ScalarDef)) {
      // Preserve pressure space for the splat
      MaxPressure[RISCV::VM] += TRI->getRegClassWeight(DstRC).RegWeight;
      if (!tryHoistSrcDef(Loop, MaxPressure, ScalarDef))
        continue;
    }

    // Insert PseudoVMV
    auto &MVInst = TII->get(getMVInst(RISCVII::getLMul(MI->getDesc().TSFlags),
                                      MRI->getRegClass(ScalarReg)));

    MachineInstr *InsertPos;
    MachineBasicBlock *InsertBB;
    InsertBB = Loop->getLoopPreheader();
    InsertPos = &InsertBB->back();

    // Create vreg for storing the splat result
    Register SplatReg = MRI->createVirtualRegister(DstRC);
    MachineInstr *New =
        BuildMI(*InsertBB, *InsertPos, MIMetadata(*InsertPos), MVInst, SplatReg)
            .addReg(MCRegister::NoRegister)
            .addReg(ScalarReg)
            .addImm(RISCV::VLMaxSentinel) // AVL
            .add(MI->getOperand(RISCVII::getSEWOpNum(MI->getDesc())))
            .addImm(RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC);
    LLVM_DEBUG(dbgs() << "Inserted VMV: ");
    LLVM_DEBUG(New->dump());

    // Expand .vx into .vv
    LLVM_DEBUG(dbgs() << "Before expansion: ");
    LLVM_DEBUG(MI->dump());
    const RISCV::RISCVVXPseudoInfo *I = RISCV::getVXPseudoInfo(MI->getOpcode());
    MI->setDesc(TII->get(I->VVPseudo));
    MI->getOperand(ScalarRegIdx).setReg(SplatReg);
    LLVM_DEBUG(dbgs() << "After expansion: ");
    LLVM_DEBUG(MI->dump());
    HoistedCount += 1;

    MRI->verifyUseList(SplatReg);
  }
  return HoistedCount;
}

bool RISCVExpandVXPseudo::isHoistable(
    const MachineLoop *Loop, llvm::DenseMap<unsigned, int> &DeltaPressure,
    MachineInstr *MI, int Depth) {
  // Terminates if its def is out of loop or recursion depth reaches the
  // pre-defined limit
  if (!Loop->contains(MI))
    return true;
  if (Depth > DepthLimit)
    return false;

  auto IsMOHoistable = [&](MachineOperand &MO) {
    if (MO.isReg()) {
      auto Reg = MO.getReg();
      // We are still incapable dealing with source with multiple defs
      if (MRI->hasOneDef(Reg))
        return isHoistable(Loop, DeltaPressure, MRI->getVRegDef(Reg),
                           Depth + 1);
      return false;
    }
    return true;
  };

  // Recursive call on all explicit_uses and and-fold the results
  bool IsHoistable = std::transform_reduce(MI->explicit_uses().begin(),
                                           MI->explicit_uses().end(), true,
                                           std::logical_and<>(), IsMOHoistable);

  // Do after the recursive call so we can see defs before kills
  for (auto [Class, Cost] : calcMICost(*MI))
    DeltaPressure[Class] += Cost;

  return IsHoistable;
}

void RISCVExpandVXPseudo::recursiveHoist(const MachineLoop *Loop,
                                         MachineInstr *MI) {
  if (!Loop->contains(MI))
    return;

  llvm::for_each(MI->explicit_uses(), [&](auto &MO) {
    if (MO.isReg())
      recursiveHoist(Loop, MRI->getVRegDef(MO.getReg()));
  });

  LLVM_DEBUG(dbgs() << "SrcDef is hoisted: ");
  LLVM_DEBUG(MI->dump());
  MI->moveBefore(&Loop->getLoopPreheader()->back());
}

bool RISCVExpandVXPseudo::tryHoistSrcDef(const MachineLoop *Loop,
                                         RegPressureSet &MaxPressure,
                                         MachineInstr *MI) {
  llvm::DenseMap<unsigned, int> DeltaPressure;
  RegSeen.clear();

  bool Hoistable = isHoistable(Loop, DeltaPressure, MI, 1);
  if (!Hoistable)
    return false;

  // Check if it's safe to hoist by calculating the new pressure
  for (auto &[Class, Cost] : DeltaPressure)
    if (MaxPressure[Class] + Cost >
        TRI->getRegPressureSetLimit(*MI->getMF(), Class))
      return false;

  // Hoist instruction
  recursiveHoist(Loop, MI);
  return true;
}

// FIXME: Do we have a better way to do this?
unsigned RISCVExpandVXPseudo::getMVInst(RISCVII::VLMUL Lmul, const TargetRegisterClass *RC) {
  if (RC == &RISCV::GPRRegClass) {
    switch (Lmul) {
    case RISCVII::LMUL_1:
      return RISCV::PseudoVMV_V_X_M1;
    case RISCVII::LMUL_2:
      return RISCV::PseudoVMV_V_X_M2;
    case RISCVII::LMUL_4:
      return RISCV::PseudoVMV_V_X_M4;
    case RISCVII::LMUL_8:
      return RISCV::PseudoVMV_V_X_M8;
    case RISCVII::LMUL_F2:
      return RISCV::PseudoVMV_V_X_MF2;
    case RISCVII::LMUL_F4:
      return RISCV::PseudoVMV_V_X_MF4;
    case RISCVII::LMUL_F8:
      return RISCV::PseudoVMV_V_X_MF8;
    default:
      return -1;
    }
  } else if (RC == &RISCV::FPR16RegClass) {
    switch (Lmul) {
    case RISCVII::LMUL_1:
      return RISCV::PseudoVFMV_V_FPR16_M1;
    case RISCVII::LMUL_2:
      return RISCV::PseudoVFMV_V_FPR16_M2;
    case RISCVII::LMUL_4:
      return RISCV::PseudoVFMV_V_FPR16_M4;
    case RISCVII::LMUL_8:
      return RISCV::PseudoVFMV_V_FPR16_M8;
    case RISCVII::LMUL_F2:
      return RISCV::PseudoVFMV_V_FPR16_MF2;
    case RISCVII::LMUL_F4:
      return RISCV::PseudoVFMV_V_FPR16_MF4;
    default:
      return -1;
    }
  } else if (RC == &RISCV::FPR32RegClass) {
    switch (Lmul) {
    case RISCVII::LMUL_1:
      return RISCV::PseudoVFMV_V_FPR32_M1;
    case RISCVII::LMUL_2:
      return RISCV::PseudoVFMV_V_FPR32_M2;
    case RISCVII::LMUL_4:
      return RISCV::PseudoVFMV_V_FPR32_M4;
    case RISCVII::LMUL_8:
      return RISCV::PseudoVFMV_V_FPR32_M8;
    case RISCVII::LMUL_F2:
      return RISCV::PseudoVFMV_V_FPR32_MF2;
    default:
      return -1;
    }
  } else if (RC == &RISCV::FPR64RegClass) {
    switch (Lmul) {
    case RISCVII::LMUL_1:
      return RISCV::PseudoVFMV_V_FPR64_M1;
    case RISCVII::LMUL_2:
      return RISCV::PseudoVFMV_V_FPR64_M2;
    case RISCVII::LMUL_4:
      return RISCV::PseudoVFMV_V_FPR64_M4;
    case RISCVII::LMUL_8:
      return RISCV::PseudoVFMV_V_FPR64_M8;
    default:
      return -1;
    }
  }
  return -1;
}
