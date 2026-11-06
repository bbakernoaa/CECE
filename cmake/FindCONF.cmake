# Custom FindCONF.cmake for in-tree HELM build
set(CONF_FOUND TRUE)
set(CONF_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/extern/helm/libs/conf/include")

if(NOT TARGET HELM::CONF)
  add_library(HELM::CONF INTERFACE IMPORTED)
  target_link_libraries(HELM::CONF INTERFACE conf)
endif()
