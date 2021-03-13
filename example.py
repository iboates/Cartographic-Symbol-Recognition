from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from cartosymrecog import *


inputs = {
    "map_path": "data/stantec/map0.tif",
    "map_area_gdf": gpd.GeoDataFrame.from_file("data/stantec/map_area.shp"),
    "symbols_gdf": gpd.GeoDataFrame.from_file("data/stantec/symbols.shp")
}


with CartoFigure(**inputs) as figure:

    # symbol_manager = CartoSymbolManager()
    # for _, row in inputs["symbols_gdf"].iterrows():
        # symbol_manager.add_symbol(row["label"], extract_image(figure.get_map(), row.geometry)),

    symbols = [(idx, row["label"], extract_image(figure.get_map(), row.geometry)) for idx, row in inputs["symbols_gdf"].iterrows()]

    # for _, row in inputs["symbols_gdf"].iterrows():
        # symbol = figure.get_symbol(row['label'])
        # figure.generate_training_data(10000, f"train/{row['label']}", num=10, rotate=row["rotate"])

    Path("train/image").mkdir(parents=True, exist_ok=True)
    Path("train/label").mkdir(parents=True, exist_ok=True)
    Path("valid/image").mkdir(parents=True, exist_ok=True)
    Path("valid/label").mkdir(parents=True, exist_ok=True)

    for i in tqdm(list(range(900))):
        for symbol in symbols:
            sample_name = f"{symbol[1]}_{str(i).zfill(9)}"
            figure.generate_training_image("train/image", "train/label", sample_name, symbol, 1000)

    for i in tqdm(list(range(100))):
        for symbol in symbols:
            sample_name = f"{symbol[1]}_{str(i).zfill(9)}"
            figure.generate_training_image("valid/image", "valid/label", sample_name, symbol, 1000)
