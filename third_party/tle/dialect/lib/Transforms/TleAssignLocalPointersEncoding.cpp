// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEASSIGNLOCALPOINTERSENCODING
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

// Triton shared-memory pointers use LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
constexpr StringLiteral kBarrierGroupAttr = "tle.barrier_group";
constexpr StringLiteral kTTContiguityAttr = "tt.contiguity";
constexpr StringLiteral kTTDivisibilityAttr = "tt.divisibility";
constexpr StringLiteral kTTConstancyAttr = "tt.constancy";

static Operation *peelAxisInfoCarrier(Value value) {
  llvm::DenseSet<Value> visited;
  Value current = value;
  while (current && visited.insert(current).second) {
    Operation *def = current.getDefiningOp();
    if (!def)
      break;
    if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(def)) {
      current = convert.getSrc();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      current = bcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
      current = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(def)) {
      current = reshape.getSrc();
      continue;
    }
    return def;
  }
  return current ? current.getDefiningOp() : nullptr;
}

static void copyAxisInfoAttrs(Operation *src, Operation *dst) {
  if (!src || !dst)
    return;
  auto tryCopy = [&](StringRef name) {
    if (dst->getDiscardableAttr(name))
      return;
    if (auto attr = src->getDiscardableAttr(name))
      dst->setDiscardableAttr(name, attr);
  };
  tryCopy(kTTContiguityAttr);
  tryCopy(kTTDivisibilityAttr);
  tryCopy(kTTConstancyAttr);
}

static void collectConsumerEncodings(
    Value root, llvm::SmallVectorImpl<Attribute> &loadEncodings,
    llvm::SmallVectorImpl<Attribute> &storeEncodings) {
  llvm::SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  auto enqueue = [&](Value v) {
    if (!v)
      return;
    if (!visited.insert(v).second)
      return;
    worklist.push_back(v);
  };

  enqueue(root);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (auto load = dyn_cast<triton::LoadOp>(owner)) {
        auto loadTy = dyn_cast<RankedTensorType>(load.getResult().getType());
        if (loadTy && loadTy.getEncoding())
          loadEncodings.push_back(loadTy.getEncoding());
        continue;
      }
      if (auto store = dyn_cast<triton::StoreOp>(owner)) {
        auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
        if (valueTy && valueTy.getEncoding())
          storeEncodings.push_back(valueTy.getEncoding());
        continue;
      }
      if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(owner)) {
        enqueue(convert.getResult());
        continue;
      }
      if (auto bcast = dyn_cast<triton::BroadcastOp>(owner)) {
        enqueue(bcast.getResult());
        continue;
      }
      if (auto expand = dyn_cast<triton::ExpandDimsOp>(owner)) {
        enqueue(expand.getResult());
        continue;
      }
      if (auto reshape = dyn_cast<triton::ReshapeOp>(owner)) {
        enqueue(reshape.getResult());
        continue;
      }
      if (auto remote = dyn_cast<triton::tle::RemotePointersOp>(owner)) {
        enqueue(remote.getResult());
        continue;
      }
    }
  }
}

class AssignLocalPointersEncodingPass
    : public impl::TritonTleAssignLocalPointersEncodingBase<
          AssignLocalPointersEncodingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    module.walk([&](triton::tle::LocalPointersOp op) {
      auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
      if (!tensorTy)
        return;
      auto ptrTy = dyn_cast<triton::PointerType>(tensorTy.getElementType());
      if (!ptrTy)
        return;
      bool updated = false;
      RankedTensorType updatedTensorTy = tensorTy;
      const auto desiredAddrSpace = kSharedMemoryAddressSpace;
      if (ptrTy.getAddressSpace() != desiredAddrSpace) {
        ptrTy =
            triton::PointerType::get(ptrTy.getPointeeType(), desiredAddrSpace);
        updated = true;
      }

      auto encoding = tensorTy.getEncoding();
      Attribute userEncoding;
      SmallVector<Attribute> loadConsumerEncodings;
      SmallVector<Attribute> storeConsumerEncodings;
      collectConsumerEncodings(op.getResult(), loadConsumerEncodings,
                               storeConsumerEncodings);
      auto pickConsistentEncoding =
          [](ArrayRef<Attribute> encodings) -> Attribute {
        Attribute selected;
        for (Attribute enc : encodings) {
          if (!selected)
            selected = enc;
          else if (selected != enc)
            return Attribute();
        }
        return selected;
      };
      // Pointer tensor encoding should follow load consumers first; stores can
      // be bridged via convert_layout on the value path.
      userEncoding = pickConsistentEncoding(loadConsumerEncodings);
      if (!userEncoding)
        userEncoding = pickConsistentEncoding(storeConsumerEncodings);
      if (userEncoding && userEncoding != encoding) {
        encoding = userEncoding;
        updated = true;
      }
      if (!encoding) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        int numWarps = triton::gpu::maybeLookupNumWarps(op).value_or(1);
        int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(builder);
        int numCTAs = triton::gpu::lookupNumCTAs(builder);
        encoding = triton::gpu::getDefaultBlockedEncoding(
            module.getContext(), tensorTy.getShape(), numWarps, threadsPerWarp,
            numCTAs);
        updated = true;
      }

      if (updated)
        updatedTensorTy =
            RankedTensorType::get(tensorTy.getShape(), ptrTy, encoding);

      if (updated)
        op.getResult().setType(updatedTensorTy);

      if (updated) {
        llvm::DenseSet<Value> visited;
        auto updateUserResultTypes = [&](auto &&self, Value ptrVal) -> void {
          if (!ptrVal || !visited.insert(ptrVal).second)
            return;
          auto ptrTensorTy = cast<RankedTensorType>(ptrVal.getType());
          auto ptrElemTy =
              cast<triton::PointerType>(ptrTensorTy.getElementType())
                  .getPointeeType();
          auto loadTy = RankedTensorType::get(ptrTensorTy.getShape(), ptrElemTy,
                                              ptrTensorTy.getEncoding());
          for (OpOperand &use : ptrVal.getUses()) {
            Operation *owner = use.getOwner();
            if (auto load = dyn_cast<triton::LoadOp>(owner)) {
              auto oldLoadTy =
                  dyn_cast<RankedTensorType>(load.getResult().getType());
              if (oldLoadTy != loadTy) {
                load.getResult().setType(loadTy);
                if (oldLoadTy) {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointAfter(load);
                  auto bridge = builder.create<triton::gpu::ConvertLayoutOp>(
                      load.getLoc(), oldLoadTy, load.getResult());
                  load.getResult().replaceAllUsesExcept(
                      bridge.getResult(), bridge.getOperation());
                }
              }
              continue;
            }
            if (auto store = dyn_cast<triton::StoreOp>(owner)) {
              auto valueTy =
                  dyn_cast<RankedTensorType>(store.getValue().getType());
              if (valueTy && valueTy.getEncoding() != ptrTensorTy.getEncoding()) {
                auto convertedTy = RankedTensorType::get(
                    valueTy.getShape(), valueTy.getElementType(),
                    ptrTensorTy.getEncoding());
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPoint(store);
                auto converted = builder.create<triton::gpu::ConvertLayoutOp>(
                    store.getLoc(), convertedTy, store.getValue());
                store.getValueMutable().assign(converted.getResult());
              }
              continue;
            }
            if (auto atomic = dyn_cast<triton::AtomicRMWOp>(owner)) {
              atomic.getResult().setType(loadTy);
              continue;
            }
            if (auto cas = dyn_cast<triton::AtomicCASOp>(owner)) {
              cas.getResult().setType(loadTy);
              continue;
            }
            if (auto remote = dyn_cast<triton::tle::RemotePointersOp>(owner)) {
              if (remote.getResult().getType() != ptrTensorTy)
                remote.getResult().setType(ptrTensorTy);
              self(self, remote.getResult());
              continue;
            }
          }
        };
        updateUserResultTypes(updateUserResultTypes, op.getResult());
      }

      auto desiredEncoding = updatedTensorTy.getEncoding();
      if (desiredEncoding) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        SmallVector<Value> newOperands;
        newOperands.reserve(op->getNumOperands());
        newOperands.push_back(op.getSrc());
        bool updatedOperands = false;
        for (Value operand : op.getIndices()) {
          auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
          if (!operandTy) {
            newOperands.push_back(operand);
            continue;
          }
          if (operandTy.getEncoding() == desiredEncoding) {
            newOperands.push_back(operand);
            continue;
          }
          auto convertedTy = RankedTensorType::get(operandTy.getShape(),
                                                   operandTy.getElementType(),
                                                   desiredEncoding);
          auto converted = builder.create<triton::gpu::ConvertLayoutOp>(
              op.getLoc(), convertedTy, operand);
          newOperands.push_back(converted);
          updatedOperands = true;
        }
        if (updatedOperands)
          op->setOperands(newOperands);
      }

      tagDependencyGroup(op, builder);
    });

    // remote_pointers should preserve source pointer axis properties so later
    // passes can reason about remote operands without dialect-specific visitors.
    module.walk([&](triton::tle::RemotePointersOp op) {
      Operation *srcDef = peelAxisInfoCarrier(op.getSrc());
      copyAxisInfoAttrs(srcDef, op.getOperation());
    });
  }

  void tagDependencyGroup(triton::tle::LocalPointersOp op, OpBuilder &builder) {
    auto alloc = op.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
    if (!alloc)
      return;
    auto groupAttr = alloc->getAttrOfType<IntegerAttr>(kBarrierGroupAttr);
    if (!groupAttr) {
      groupAttr = builder.getI64IntegerAttr(nextBarrierGroupId++);
      alloc->setAttr(kBarrierGroupAttr, groupAttr);
    }
    op->setAttr(kBarrierGroupAttr, groupAttr);
  }

  int64_t nextBarrierGroupId = 0;
};

} // namespace
} // namespace mlir::triton::tle
