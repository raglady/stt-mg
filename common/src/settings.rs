use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommonSettings {
    pub general: GeneralSettings,
    #[serde(flatten)]
    pub stt_settings: stt_train::settings::Settings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralSettings {
    pub end_press_key: bool,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            end_press_key: true,
        }
    }
}
