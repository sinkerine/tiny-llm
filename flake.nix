{
  description = "tiny-llm CUDA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      cuda = pkgs.cudaPackages_13_0;
      runtimeLibraries = [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          pdm
          pyright
          ruff
          direnv
          nix-direnv
          python312Packages.huggingface-hub
          cmake
          ninja
          pkg-config
          gcc
          git
          cuda.cuda_nvcc
          cuda.cudatoolkit
        ];

        env = {
          CUDA_HOME = "${cuda.cudatoolkit}";
          CUDA_PATH = "${cuda.cudatoolkit}";
          CUDACXX = "${cuda.cuda_nvcc}/bin/nvcc";
          PDM_VENV_IN_PROJECT = "1";
          PYTHONNOUSERSITE = "1";
        };

        shellHook = ''
          export HF_HUB_CACHE="$HOME/.cache/tiny-llm"
          export PATH="${cuda.cuda_nvcc}/bin:${cuda.cudatoolkit}/bin:$PATH"
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibraries}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          if [ -x .venv/bin/python ]; then
            export VIRTUAL_ENV="$PWD/.venv"
            export PATH="$VIRTUAL_ENV/bin:$PATH"
          fi
        '';
      };
    };
}
