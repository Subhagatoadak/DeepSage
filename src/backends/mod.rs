pub mod llamacpp;
pub mod ollama;

#[derive(Debug, Clone)]
pub struct RunningModel {
    pub name: String,
    pub backend: String,
    #[allow(dead_code)] pub pid: Option<u32>,
    #[allow(dead_code)] pub vram_gb: f32,
    #[allow(dead_code)] pub ram_gb: f32,
    pub endpoint: Option<String>,
}

#[allow(dead_code)]
pub trait Backend {
    fn name(&self) -> &'static str;
}
