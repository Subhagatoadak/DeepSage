use sysinfo::System;

#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    pub cpu_pct: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub swap_used_gb: f32,
    pub swap_total_gb: f32,
}

impl ResourceStats {
    pub fn ram_pct(&self) -> f32 {
        if self.ram_total_gb > 0.0 {
            self.ram_used_gb / self.ram_total_gb
        } else {
            0.0
        }
    }
    pub fn swap_pct(&self) -> f32 {
        if self.swap_total_gb > 0.0 {
            self.swap_used_gb / self.swap_total_gb
        } else {
            0.0
        }
    }
}

pub fn collect() -> ResourceStats {
    let mut sys = System::new_all();
    sys.refresh_all();

    let gb = |b: u64| b as f32 / 1024.0 / 1024.0 / 1024.0;

    ResourceStats {
        cpu_pct: sys.global_cpu_usage(),
        ram_used_gb: gb(sys.used_memory()),
        ram_total_gb: gb(sys.total_memory()),
        swap_used_gb: gb(sys.used_swap()),
        swap_total_gb: gb(sys.total_swap()),
    }
}
