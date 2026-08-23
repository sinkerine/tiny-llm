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
          pdm
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
          export PATH="${cuda.cuda_nvcc}/bin:${cuda.cudatoolkit}/bin:$PATH"
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibraries}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
