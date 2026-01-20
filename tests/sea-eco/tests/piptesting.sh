rm -rf test sea-eco             # purge previous run folders
uv venv test                    # create venv
source test/bin/activate        # activate it
git clone https://tpchuckles:ghp_abc123thisisafakekeyobviouslylol@github.com/sea-ecosystem/sea-eco
uv pip install -e sea-eco       # test pip install
git clone https://tpchuckles:ghp_abc123thisisafakekeyobviouslylol@github.com/sea-ecosystem/rayTEM
uv pip install -e rayTEM
python3 -c "from pySEA.rayTEM.elements import Lens; print('INSTALL SUCCESSFUL')"
