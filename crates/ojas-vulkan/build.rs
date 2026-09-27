//! Compile every `shaders/*.comp` to SPIR-V at build time (glslang, the
//! Khronos reference compiler; nothing compiles at run time). Each shader is
//! built once per precision it declares on its first line:
//! `// precisions: f32 f16` → entries `<stem>` (float storage) and
//! `<stem>_f16` (`-DF16`, float16_t storage); `// precisions: f16` → only
//! `<stem>_f16`. `shaders/prelude.glsl` is prepended to every shader.
//! Output: `$OUT_DIR/kernels.rs` = `KERNELS: &[(&str, &[u8])]`.

use glslang::{Compiler, CompilerOptions, ShaderInput, ShaderSource, ShaderStage, SourceLanguage, Target, VulkanVersion};
use std::fmt::Write as _;
use std::path::Path;

fn main() {
    let dir = Path::new("shaders");
    println!("cargo:rerun-if-changed=shaders");
    let out = std::env::var("OUT_DIR").unwrap();
    let prelude = std::fs::read_to_string(dir.join("prelude.glsl")).expect("shaders/prelude.glsl");
    let compiler = Compiler::acquire().expect("glslang init");
    let opts = CompilerOptions {
        source_language: SourceLanguage::GLSL,
        // SPIR-V 1.5: 1.6 declares spec-constant workgroup sizes with OpExecutionModeId,
        // which the 1.3.204 validation layer (Ubuntu 22.04) cannot parse
        target: Target::Vulkan { version: VulkanVersion::Vulkan1_3, spirv_version: glslang::SpirvVersion::SPIRV1_5 },
        version_profile: None,
        messages: glslang::ShaderMessage::DEFAULT,
    };
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "comp")).collect();
    files.sort();
    let mut table = String::from("pub static KERNELS: &[(&str, &[u8])] = &[\n");
    for path in files {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let body = std::fs::read_to_string(&path).unwrap();
        let first = body.lines().next().unwrap_or("");
        let precisions: Vec<&str> = first.strip_prefix("// precisions:").map(|s| s.split_whitespace().collect()).unwrap_or_else(|| vec!["f32", "f16"]);
        // `// name: x` on the second line: the entry name of a single-precision kernel
        let fixed = body.lines().nth(1).and_then(|l| l.strip_prefix("// name:")).map(|n| n.trim().to_string());
        for prec in precisions {
            let name = match &fixed {
                Some(n) => n.clone(),
                None if prec == "f16" => format!("{stem}_f16"),
                None => stem.clone(),
            };
            let defines: Vec<(&str, Option<&str>)> = if prec == "f16" { vec![("F16", None)] } else { vec![] };
            // #line keeps compiler messages pointing at the shader file's own lines
            let src = format!("{prelude}\n#line 1\n{body}");
            let source = ShaderSource::from(src);
            let input = ShaderInput::new(&source, ShaderStage::Compute, &opts, Some(&defines[..]), None).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let shader = compiler.create_shader(input).unwrap_or_else(|e| panic!("{name}: {e}"));
            let words = shader.compile().unwrap_or_else(|e| panic!("{name}: {e}"));
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let file = format!("{name}.spv");
            std::fs::write(Path::new(&out).join(&file), bytes).unwrap();
            writeln!(table, "    (\"{name}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\"))),").unwrap();
        }
    }
    table.push_str("];\n");
    std::fs::write(Path::new(&out).join("kernels.rs"), table).unwrap();
}
