class Deepsage < Formula
  desc "Configure, run, manage, and monitor open source LLM models"
  homepage "https://github.com/subhagatoadak/DeepSage"
  license "MIT"
  version "0.1.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/subhagatoadak/DeepSage/releases/download/v#{version}/deepsage-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER_AARCH64_MACOS"
    else
      url "https://github.com/subhagatoadak/DeepSage/releases/download/v#{version}/deepsage-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER_X86_64_MACOS"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/subhagatoadak/DeepSage/releases/download/v#{version}/deepsage-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER_AARCH64_LINUX"
    else
      url "https://github.com/subhagatoadak/DeepSage/releases/download/v#{version}/deepsage-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER_X86_64_LINUX"
    end
  end

  # llmfit handles hardware detection — install it alongside deepsage
  depends_on "llmfit"

  def install
    bin.install "deepsage"
  end

  def caveats
    <<~EOS
      DeepSage stores config and model data in:
        #{Dir.home}/.config/deepsage/

      To get started:
        deepsage system        # detect your hardware
        deepsage recommend     # see which models fit
        deepsage               # launch TUI dashboard
    EOS
  end

  test do
    system "#{bin}/deepsage", "--version"
  end
end
