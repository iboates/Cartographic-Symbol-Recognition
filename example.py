import geopandas as gpd

from cartosymrecog import CartoFigure


inputs = {
    "map_path": "data/stantec/map0.tif",
    "map_area_gdf": gpd.GeoDataFrame.from_file("data/stantec/map_area.shp"),
    "symbols_gdf": gpd.GeoDataFrame.from_file("data/stantec/symbols.shp")
}
with CartoFigure(**inputs) as figure:
    for _, row in inputs["symbols_gdf"].iterrows():
        symbol = figure.get_symbol(row['label'])
        figure.generate_training_images(symbol, 10000, f"train/{row['label']}", num=10, rotate=row["rotate"])
