from random import uniform, randint

import numpy as np
from PIL import Image
import rasterio as rio
from rasterio import MemoryFile
from rasterio.mask import mask, raster_geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import Point


def extract_image(rst, polygon):

    with MemoryFile() as memfile:

        meta = rst.meta.copy()
        meta["count"] = 4

        rgb = mask(rst, [polygon])[0]
        a = raster_geometry_mask(rst, [polygon], invert=True)[0].astype(rio.uint8)
        a = np.where(a == 1, 255, 0).astype(rio.uint8)
        img_data = np.stack((rgb[0], rgb[1], rgb[2], a))

        with memfile.open(**meta) as masked:
            masked.write(img_data)

            r = masked.read(1, window=from_bounds(*polygon.bounds, rst.transform))
            g = masked.read(2, window=from_bounds(*polygon.bounds, rst.transform))
            b = masked.read(3, window=from_bounds(*polygon.bounds, rst.transform))
            a = masked.read(4, window=from_bounds(*polygon.bounds, rst.transform))

    img = Image.fromarray(np.dstack((r, g, b, a)))
    return img


def random_squares_in_polygon(num, polygon, side_length):
    squares = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(squares) < num:
        pnt = Point(uniform(minx, maxx), uniform(miny, maxy))
        if polygon.contains(pnt):
            squares.append(pnt.buffer(0.5*side_length).envelope)
    return squares


def stamp_symbol(background, symbol, side_length_px, rotate):

    if rotate:
        symbol = symbol.rotate(randint(0, 360), resample=Image.BICUBIC)
    symbol_width, symbol_height = symbol.size
    top_left = (randint(0, int(side_length_px - symbol_width)), randint(0, int(side_length_px - symbol_height)))
    background.paste(symbol, top_left, symbol)
    return background
