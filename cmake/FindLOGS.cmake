# Custom FindLOGS.cmake for in-tree HELM build
set(LOGS_FOUND TRUE)
set(LOGS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/extern/helm/libs/logs/include")

if(NOT TARGET HELM::LOGS)
  add_library(HELM::LOGS INTERFACE IMPORTED)
  target_link_libraries(HELM::LOGS INTERFACE logs)
endif()
