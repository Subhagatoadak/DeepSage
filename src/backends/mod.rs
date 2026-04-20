pub mod llamacpp;
pub mod ollama;

#[derive(Debug, Clone)]
pub struct RunningModel {
    pub name: String,
    pub backend: String,
    pub pid: Option<u32>,
    pub vram_gb: f32,
    pub ram_gb: f32,
    pub endpoint: Option<String>,
}

pub trait Backend {
    fn name(&self) -> &'static str;
}
