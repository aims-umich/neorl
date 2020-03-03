# -*- mode: python -*-

block_cipher = None


a = Analysis(['neorl.py'],
             pathex=['.','./src','./src/rl','./src/evolu','./src/utils', '/home/majdi/anaconda3/lib/python3.6/site-packages/gym/'],
             binaries=[],
             datas=[],
             hiddenimports=['cython'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='neorl',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
