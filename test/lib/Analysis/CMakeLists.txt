if(FLAGTREE_BACKEND)
  set(TestMembarCpp PARTIAL_SOURCES_INTENDED)
else()
  set(TestMembarCpp TestMembar.cpp)
endif()

add_mlir_library(TritonTestAnalysis
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  ${TestMembarCpp}

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
