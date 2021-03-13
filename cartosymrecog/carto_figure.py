from pathlib import Path
from collections import OrderedDict
from uuid import uuid4

import rasterio as rio
import geopandas as gpd
from tqdm import tqdm

from cartosymrecog.exceptions import SymbolNotFoundError
from cartosymrecog.utils import *


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

        symbols = OrderedDict()
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


    def generate_training_image(self, image_folder, label_folder, sample_name, symbol, side_length, rotate=False):

        """
        Generates a training image with an associated label file.

        symbol = (index, name, image)
        """

        square = random_squares_in_polygon(1, self._map_area.buffer(side_length * -1.41 * 0.5), side_length)[0]
        side_length_px = side_length / self._m_per_px
        with extract_image(self._map, square) as training_img:
            top_left, center = get_stamp_coordinates(symbol[2], side_length_px)
            center_norm = center[0] / training_img.size[0], center[1] / training_img.size[1]
            width, height = symbol[2].size[0] / training_img.size[0], symbol[2].size[1] / training_img.size[1]
            stamp_symbol(training_img, symbol[2], top_left, rotate)
            training_img.save(f"{image_folder}/{sample_name}.png")
            with open(f"{label_folder}/{sample_name}.txt", "w") as f:
                f.write(f"{symbol[0]} {center_norm[0]} {center_norm[1]} {width} {height}")


    # def generate_training_data(self, side_length, out_folder, num=1, rotate=False):
    #
    #     symbols_as_list = list(self._symbols)
    #     side_length_px = side_length / self._m_per_px
    #
    #     Path(out_folder, "image").mkdir(parents=True, exist_ok=True)
    #     Path(out_folder, "label").mkdir(parents=True, exist_ok=True)
    #
    #     squares = random_squares_in_polygon(num, self._map_area.buffer(side_length*-1.41*0.5), side_length)
    #     for i, square in tqdm(enumerate(squares)):
    #         with extract_image(self._map, square) as training_img:
    #             for symbol_name in self._symbols:
    #                 # stamp_symbol(training_img, symbol, side_length_px, rotate)
    #                 top_left, center = get_stamp_coordinates(self._symbols["symbol_name"], side_length_px)
    #                 stamp_symbol(training_img, self._symbols["symbol_name"], top_left, rotate)
    #                 training_img.save(f"{out_folder}/image/{str(i).zfill(9)}.png")
    #
    #                 center_norm = center[0] / training_img.size[0], center[1] / training_img.size[1]
    #                 width, height = self._symbols["symbol_name"].size[0] / training_img.size[0], self._symbols["symbol_name"].size[1] / training_img.size[1]
    #
    #                 with open(f"{out_folder}/label/{str(i).zfill(9)}.txt", "w") as f:
    #                     f.write(f"{symbols_as_list.index(symbol)} {center_norm[0]} {center_norm[1]} {width} {height}")
