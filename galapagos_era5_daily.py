#!/usr/bin/env python
# coding: utf-8

# # 🌊 ERA5 Galápagos — Datos diarios por evento climático
# 
# Descarga **1 request por evento** con datos diarios (solo hora 12:00 UTC).
# 
# Cada request cubre únicamente los meses del evento ± margen → archivos pequeños y sin error 403.
# 
# | Evento | Período | Meses |
# |---|---|---|
# | El Niño 1982-83 | Sep 1982 – Jun 1983 | 10 meses |
# | El Niño 1997-98 | May 1997 – Jul 1998 | 15 meses |
# | El Niño 2015-16 | Mar 2015 – Jun 2016 | 16 meses |
# | La Niña 2010-11 | Jun 2010 – Mar 2011 | 10 meses |
# | Inundaciones 2023 | Oct 2023 – Feb 2024 | 5 meses |

# In[2]:


import os, glob, warnings
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
os.environ['CDSAPI_KEY'] = '30eb862e-b956-40a3-a405-1def0d04b8fd'  # ← tu key

os.makedirs('galapagos_era5/nc',  exist_ok=True)
os.makedirs('galapagos_era5/csv', exist_ok=True)

client = cdsapi.Client()
print('✅ Conexión lista')


# In[3]:


# ── Parámetros fijos ────────────────────────────────────────────
BBOX = [2, -93, -2, -88]  # [N, W, S, E] Galápagos

VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
    'surface_pressure',
    'sea_surface_temperature',
    'surface_solar_radiation_downwards',
    'mean_sea_level_pressure',
]

# Solo mediodía UTC → 1 snapshot diario, suficiente para tendencias
HOURS = ['12:00']
DAYS  = [f'{d:02d}' for d in range(1, 32)]

# Eventos: cada uno con sus años y meses exactos del período crítico
# Formato: { 'key': {'label': str, 'years': [str], 'months': [str]} }
EVENTS = {
    'el_nino_1982_83': {
        'label':  'El Niño 1982-83',
        'years':  ['1982', '1983'],
        'months': ['09','10','11','12', '01','02','03','04','05','06'],  # Sep82–Jun83
        # ⚠️  Como cruza años, se hace en 2 requests (ver celda siguiente)
        'periods': [
            {'years': ['1982'], 'months': ['09','10','11','12']},
            {'years': ['1983'], 'months': ['01','02','03','04','05','06']},
        ]
    },
    'el_nino_1997_98': {
        'label': 'El Niño 1997-98',
        'periods': [
            {'years': ['1997'], 'months': ['05','06','07','08','09','10','11','12']},
            {'years': ['1998'], 'months': ['01','02','03','04','05','06','07']},
        ]
    },
    'el_nino_2015_16': {
        'label': 'El Niño 2015-16',
        'periods': [
            {'years': ['2015'], 'months': ['03','04','05','06','07','08','09','10','11','12']},
            {'years': ['2016'], 'months': ['01','02','03','04','05','06']},
        ]
    },
    'la_nina_2010_11': {
        'label': 'La Niña 2010-11',
        'periods': [
            {'years': ['2010'], 'months': ['06','07','08','09','10','11','12']},
            {'years': ['2011'], 'months': ['01','02','03','04']},
        ]
    },
    'floods_2023_24': {
        'label': 'Inundaciones 2023-24',
        'periods': [
            {'years': ['2023'], 'months': ['10','11','12']},
            {'years': ['2024'], 'months': ['01','02']},
        ]
    },
}

# Calcula total de requests
total = sum(len(e['periods']) for e in EVENTS.values())
print(f'Total requests a enviar: {total} (uno por período de cada evento)')


# In[5]:


# ── Descarga ─────────────────────────────────────────────────────
failed = []

for event_key, event in EVENTS.items():
    label = event['label']
    for i, period in enumerate(event['periods']):
        out = f'galapagos_era5/nc/{event_key}_p{i+1}.nc'

        if os.path.exists(out):
            print(f'⏭️  Ya existe: {out}')
            continue

        yr = period['years'][0]
        m0, m1 = period['months'][0], period['months'][-1]
        print(f'⬇️  {label} | {yr} meses {m0}–{m1} ...', end=' ', flush=True)

        try:
            client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable':      VARIABLES,
                    'year':          period['years'],
                    'month':         period['months'],
                    'day':           DAYS,
                    'time':          HOURS,
                    'area':          BBOX,
                    'format':        'netcdf',
                },
                out
            )
            print('✅')
        except Exception as e:
            print(f'❌  {e}')
            failed.append(out)

if failed:
    print(f'\n⚠️  Fallaron {len(failed)} requests: {failed}')
else:
    print('\n🎉 Todas las descargas completadas')


# In[10]:


# ── Descomprimir archivos ZIP entregados por Copernicus ──────────
import zipfile, glob, os

nc_files = sorted(glob.glob('galapagos_era5/nc/*.nc'))

for zpath in nc_files:
    # Verificar que es ZIP
    with open(zpath, 'rb') as f:
        if f.read(4) != b'PK\x03\x04':
            continue  # ya es NetCDF real, saltar

    extract_dir = zpath.replace('.nc', '_extracted')
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zpath, 'r') as zf:
        print(f'📦 {os.path.basename(zpath)} contiene: {zf.namelist()}')
        zf.extractall(extract_dir)

    print(f'   ✅ Extraído en: {extract_dir}')

print('\n🎉 Descompresión completa')


# In[11]:


# ── Combinar extraídos y exportar CSV por evento ─────────────────
import glob, os
import xarray as xr
import pandas as pd
import numpy as np

for event_key, event in EVENTS.items():
    extract_dirs = sorted(glob.glob(f'galapagos_era5/nc/{event_key}_*_extracted'))
    if not extract_dirs:
        print(f'⚠️  Sin carpetas extraídas: {event_key}')
        continue

    dfs = []
    for d in extract_dirs:
        f_instant = os.path.join(d, 'data_stream-oper_stepType-instant.nc')
        f_accum   = os.path.join(d, 'data_stream-oper_stepType-accum.nc')

        ds_i = xr.open_dataset(f_instant, engine='netcdf4')
        ds_a = xr.open_dataset(f_accum,   engine='netcdf4')

        df_i = ds_i.mean(dim=['latitude','longitude']).to_dataframe()
        df_a = ds_a.mean(dim=['latitude','longitude']).to_dataframe()
        ds_i.close(); ds_a.close()

        # Unir instant + accum por índice de tiempo
        df_chunk = df_i.join(df_a, how='outer', lsuffix='', rsuffix='_accum')
        dfs.append(df_chunk)

    df = pd.concat(dfs).sort_index()

    # Conversiones de unidades
    for col, name in [('t2m','t2m_c'), ('d2m','d2m_c'), ('sst','sst_c')]:
        if col in df.columns: df[name] = df[col] - 273.15
    for col, name in [('sp','sp_hpa'), ('msl','msl_hpa')]:
        if col in df.columns: df[name] = df[col] / 100
    if 'u10' in df.columns and 'v10' in df.columns:
        df['wind_speed_ms'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['wind_dir_deg']  = (270 - np.degrees(np.arctan2(df['v10'], df['u10']))) % 360

    out_csv = f'galapagos_era5/csv/{event_key}.csv'
    df.to_csv(out_csv)
    print(f'✅ {event["label"]:30s} → {out_csv}  ({len(df)} filas)')

print('\n🎉 CSVs listos')


# In[12]:


# ── Ver columnas y muestra de datos ─────────────────────────────
import pandas as pd

for event_key, event in EVENTS.items():
    df = pd.read_csv(f'galapagos_era5/csv/{event_key}.csv', index_col=0, parse_dates=True)
    print(f'\n📊 {event["label"]} ({len(df)} filas)')
    print(f'   Período: {df.index.min().date()} → {df.index.max().date()}')
    print(f'   Columnas: {list(df.columns)}')
    print(df[['t2m_c','sst_c','wind_speed_ms']].describe().round(2))


# In[13]:


# ── Visualización comparativa de todos los eventos ───────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

EVENTOS_PLOT = {
    'el_nino_1982_83': 'El Niño 1982-83',
    'el_nino_1997_98': 'El Niño 1997-98',
    'el_nino_2015_16': 'El Niño 2015-16',
    'la_nina_2010_11': 'La Niña 2010-11',
    'floods_2023_24':  'Inundaciones 2023-24',
}

COLS = [
    ('t2m_c',        'Temp 2m (°C)',   'tomato'),
    ('sst_c',        'SST (°C)',        'steelblue'),
    ('wind_speed_ms','Viento (m/s)',    'seagreen'),
    ('tp',           'Precip (m/día)', 'navy'),
]

fig, axes = plt.subplots(len(EVENTOS_PLOT), 4, figsize=(22, 4*len(EVENTOS_PLOT)))
fig.suptitle('ERA5 — Eventos Climáticos Críticos en Galápagos', fontsize=14, fontweight='bold', y=1.01)

for row, (key, label) in enumerate(EVENTOS_PLOT.items()):
    df = pd.read_csv(f'galapagos_era5/csv/{key}.csv', index_col=0, parse_dates=True)
    # Eliminar columnas de metadatos
    df = df.drop(columns=[c for c in ['number','expver','number_accum','expver_accum'] if c in df.columns])

    for col_idx, (col, ylabel, color) in enumerate(COLS):
        ax = axes[row][col_idx]
        if col in df.columns:
            ax.plot(df.index, df[col], color=color, linewidth=0.8, alpha=0.85)
            # Línea de media
            ax.axhline(df[col].mean(), color=color, linewidth=1.2, linestyle='--', alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
        ax.grid(True, alpha=0.2)
        if col_idx == 0:
            ax.set_title(label, fontsize=10, fontweight='bold', loc='left', pad=4)

plt.tight_layout()
os.makedirs('galapagos_era5/csv', exist_ok=True)
plt.savefig('galapagos_era5/csv/todos_eventos.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Figura guardada en galapagos_era5/csv/todos_eventos.png')


# In[14]:


import pandas as pd

umbrales = {'fria': 23, 'alerta': 26, 'critica': 28}

for event_key, label in EVENTOS_PLOT.items():
    df = pd.read_csv(f'galapagos_era5/csv/{event_key}.csv', index_col=0, parse_dates=True)
    total = len(df)
    print(f'\n🐟 {label}')
    print(f'   SST < 23°C (peces pelágicos):  {(df.sst_c < 23).sum():3d} días ({(df.sst_c < 23).mean()*100:.0f}%)')
    print(f'   SST > 26°C (alerta migración): {(df.sst_c > 26).sum():3d} días ({(df.sst_c > 26).mean()*100:.0f}%)')
    print(f'   SST > 28°C (colapso crítico):  {(df.sst_c > 28).sum():3d} días ({(df.sst_c > 28).mean()*100:.0f}%)')


# In[ ]:




