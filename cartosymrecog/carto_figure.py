from pathlib import Path
from uuid import uuid4

import rasterio as rio
import geopandas as gpd
from tqdm import tqdm

from cartosymrecog.exceptions import SymbolNotFoundError
from cartosymrecog.utils import extract_image, random_squares_in_polygon, stamp_symbol


class CartoFigure:

    def __init__(self, map_path: Path, map_area_gdf: gpd.GeoDataFrame, symbols_gdf: gpd.GeoDataFrame, symbol_label_col="label"):

        self._map = rio.open(map_path)
        self._m_per_px = self._map.meta["transform"][0]
        self._symbols = self._load_symbols(symbols_gdf, symbol_label_col)

        self._map_area = map_area_gdf.iloc[0].geometry

    def __enter__(self):

        return self

    def __exit__(self, *args):

        self.close()

    def _load_symbols(self, symbols_gdf, symbol_label_col):

        symbols = {}
        for _, row in symbols_gdf.iterrows():
            symbols[row[symbol_label_col]] = extract_image(self._map, row.geometry)
        return symbols

    def close(self):

        self._map.close()

    def get_symbol(self, name):

        try:
            return self._symbols[name]
        except KeyError:
            SymbolNotFoundError(f'Symbol {name} not found.')

    def get_map(self):

        return self._map

    def generate_training_images(self, symbols, side_length, out_folder, num=1, rotate=False):

        side_length_px = side_length / self._m_per_px

        try:
            iter(symbols)
        except TypeError:
            symbols = [symbols]

        Path(out_folder).mkdir(parents=True, exist_ok=True)

        squares = random_squares_in_polygon(num, self._map_area.buffer(side_length*-1.41*0.5), side_length)
        for square in tqdm(squares):
            with extract_image(self._map, square) as training_img:
                for symbol in symbols:
                    stamp_symbol(training_img, symbol, side_length_px, rotate)
                training_img.save(f"{out_folder}/{uuid4()}.png")
