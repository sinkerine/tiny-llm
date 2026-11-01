{
  description = "Tiny LLM Week 1: PyTorch CUDA development";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ python312 uv pdm direnv nix-direnv pyright ruff ];
        PDM_VENV_IN_PROJECT = "1";
        PYTHONNOUSERSITE = "1";
        UV_PYTHON_DOWNLOADS = "never";
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export PYTHONPATH="$PWD/src:$PWD''${PYTHONPATH:+:$PYTHONPATH}"
          if [ ! -x .venv/bin/python ]; then
            uv venv --python ${pkgs.python312}/bin/python .venv
          fi
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$VIRTUAL_ENV/bin:$PATH"
          unset PYTHONHOME
          unset NIX_PYTHONPATH
          export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
        '';
      };
    };
}
