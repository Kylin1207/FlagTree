#include "tle/dialect/include/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "tle/dialect/include/IR/Dialect.h"

namespace {

using namespace mlir;
namespace tle = mlir::triton::tle;

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DistributedBarrierOp> {
  using ConvertOpToLLVMPattern<tle::DistributedBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto unit = UnitAttr::get(ctx);
    rewriter.create<NVVM::ClusterArriveOp>(op.getLoc(), unit);
    rewriter.create<NVVM::ClusterWaitOp>(op.getLoc(), unit);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void tle::populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit);
}
