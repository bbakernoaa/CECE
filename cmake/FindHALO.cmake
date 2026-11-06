# Custom FindHALO.cmake for in-tree HELM build
set(HALO_FOUND TRUE)
set(HALO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/extern/helm/libs/halo/include")

if(NOT TARGET HELM::HALO)
  add_library(HELM::HALO INTERFACE IMPORTED)
  target_link_libraries(HELM::HALO INTERFACE halo)
endif()
