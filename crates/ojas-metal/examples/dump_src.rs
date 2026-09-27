fn main() {
    for fam in ["gemv", "requant_iq", "ops"] {
        let s = ojas_metal::kernels::family_source(fam).unwrap();
        std::fs::write(format!("/tmp/src_{fam}.metal"), s).unwrap();
        println!("{fam}: {} bytes", s.len());
    }
}
