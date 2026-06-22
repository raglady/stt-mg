use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Configuration file (--config-file)
    #[arg(long)]
    pub config_file: Option<String>,
}
