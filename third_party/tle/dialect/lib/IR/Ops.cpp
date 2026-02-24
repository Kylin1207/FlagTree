#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
namespace mlir::triton::tle {

/*void ExtractTileOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState, Value input,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<Type> tys = {
      RankedTensorType::get(tileShape, inputType.getElementType())};
  build(odsBuilder, odsState, tys, input, index);
}*/

LogicalResult DSLRegionOp::verify() {
  Region &body = getBody();
  const uint32_t numArguments = body.getNumArguments(),
                 numOperands = getNumOperands();
  if (numArguments != numOperands) {
    return emitOpError() << "expects number of operands (" << numArguments
                         << ") to match number of region arguments ("
                         << numOperands << ")";
  }
  for (auto [arg, operand] : llvm::zip(body.getArguments(), getOperands())) {
    if (arg.getType() != operand.getType()) {
      return emitOpError() << "expects region argument type (" << arg.getType()
                           << ") to match operand type (" << operand.getType()
                           << ")";
    }
  }
  return success();
}

void ExtractSizesOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, size_t num,
                           Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

void ExtractStridesOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState, size_t num,
                             Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

LogicalResult PackOp::verify() {
  TypedValue<LLVM::LLVMStructType> input = getInput();
  ArrayRef<Type> body = input.getType().getBody();
  if (body.size() < 3 || body.size() % 2 != 1 ||
      !isa<LLVM::LLVMPointerType>(body[0]) ||
      !isa<LLVM::LLVMPointerType>(body[1])) {
    return emitOpError() << "expects input struct to have at least 3 elements, "
                            "with the first two being pointer types.";
  }
  return success();
}

// ============================================================================
// 辅助函数：获取 CTA tile shape
// ============================================================================
static SmallVector<int64_t> getShapePerCTATile(RankedTensorType type) {
  auto encoding = type.getEncoding();
  auto shape = type.getShape();

  if (auto blocked = dyn_cast<gpu::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();

    SmallVector<int64_t> ctaTileShape;
    for (size_t i = 0; i < shape.size(); ++i) {
      ctaTileShape.push_back(static_cast<int64_t>(sizePerThread[i]) *
                             static_cast<int64_t>(threadsPerWarp[i]) *
                             static_cast<int64_t>(warpsPerCTA[i]));
    }
    return ctaTileShape;
  }

  if (auto linear = dyn_cast<gpu::LinearEncodingAttr>(encoding)) {
    auto sizePerThread = linear.getSizePerThread();
    auto threadsPerWarp = linear.getThreadsPerWarp();
    auto warpsPerCTA = linear.getWarpsPerCTA();

    SmallVector<int64_t> ctaTileShape;
    for (size_t i = 0; i < shape.size(); ++i) {
      ctaTileShape.push_back(static_cast<int64_t>(sizePerThread[i]) *
                             static_cast<int64_t>(threadsPerWarp[i]) *
                             static_cast<int64_t>(warpsPerCTA[i]));
    }
    return ctaTileShape;
  }

  llvm_unreachable("Unsupported encoding for extract_tile");
}

// ============================================================================
// ExtractTileOp Builder
// ============================================================================
void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          ArrayRef<int64_t> static_offsets,
                          ArrayRef<int64_t> tileShape) {

  auto srcType = cast<RankedTensorType>(src.getType());

  auto resultType = RankedTensorType::get(tileShape, srcType.getElementType(),
                                          srcType.getEncoding());

  state.addOperands(src);
  state.addAttribute("static_offsets",
                     builder.getDenseI64ArrayAttr(static_offsets));
  state.addTypes(resultType);
}

// ============================================================================
// ExtractTileOp Verification
// ============================================================================
LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());

  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto offsets = getStaticOffsets();

  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitError("result element type must match source element type");
  }

  if (srcTy.getRank() != dstTy.getRank()) {
    return emitError("result rank must equal source rank");
  }

  if (offsets.size() != static_cast<size_t>(srcTy.getRank())) {
    return emitError("offsets size must match tensor rank");
  }

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] > srcShape[i]) {
      return emitOpError(
                 "result shape cannot exceed source shape at dimension ")
             << i;
    }

    if (offsets[i] + dstShape[i] > srcShape[i]) {
      return emitOpError("invalid offset at dimension ")
             << i << ": offset(" << offsets[i] << ") + shape(" << dstShape[i]
             << ") > source(" << srcShape[i] << ")";
    }

    if (offsets[i] < 0) {
      return emitOpError("offset must be non-negative at dimension ") << i;
    }
  }

  auto encoding = srcTy.getEncoding();

  if (!encoding) {
    return success();
  }
  if (auto blocked = dyn_cast_or_null<gpu::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();

    SmallVector<int64_t> ctaTileShape;
    for (size_t i = 0; i < srcShape.size(); ++i) {
      ctaTileShape.push_back(static_cast<int64_t>(sizePerThread[i]) *
                             static_cast<int64_t>(threadsPerWarp[i]) *
                             static_cast<int64_t>(warpsPerCTA[i]));
    }

    for (size_t i = 0; i < srcShape.size(); ++i) {
      if (offsets[i] % ctaTileShape[i] != 0) {
        return emitOpError(
                   "offset must be multiple of CTA tile size at dimension ")
               << i << " (got " << offsets[i] << ", expected multiple of "
               << ctaTileShape[i] << ")";
      }

      if (dstShape[i] % ctaTileShape[i] != 0) {
        return emitOpError("result shape must be multiple of CTA tile size at "
                           "dimension ")
               << i << " (got " << dstShape[i] << ", expected multiple of "
               << ctaTileShape[i] << ")";
      }
    }
  }

  return success();
}

} // namespace mlir::triton::tle
