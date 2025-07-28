set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STAGE_VERT -fshader-stage=vert)
set(STAGE_FRAG -fshader-stage=frag)
set(GRAPHICS_VERT_SHADER ${SHADER_DIR}/graphics.vert.glsl)
set(GRAPHICS_FRAG_SHADER ${SHADER_DIR}/graphics.frag.glsl)
set(GRAPHICS_SPIRV_VERT ${SHADER_DIR}/graphics.vert.spv)
set(GRAPHICS_SPIRV_FRAG ${SHADER_DIR}/graphics.frag.spv)

# glsl -> SPIR-V compilation

add_custom_command(
     OUTPUT ${GRAPHICS_SPIRV_VERT}
     COMMAND glslc ${STAGE_VERT} ${GRAPHICS_VERT_SHADER} -o ${GRAPHICS_SPIRV_VERT}
     DEPENDS ${GRAPHICS_VERT_SHADER}
)

add_custom_command(
     OUTPUT ${GRAPHICS_SPIRV_FRAG}
     COMMAND glslc ${STAGE_FRAG} ${GRAPHICS_FRAG_SHADER} -o ${GRAPHICS_SPIRV_FRAG}
     DEPENDS ${GRAPHICS_FRAG_SHADER}
)

add_custom_target(CompileShaders ALL
    DEPENDS ${GRAPHICS_SPIRV_VERT} ${GRAPHICS_SPIRV_FRAG}
)