"""Land, country and continent at a coordinate, from Natural Earth's 1:110m land and
country polygons (public domain). The files are downloaded once into a cache
directory.

The 1:110m polygons are coarse near coasts and borders, so the tasks built on them
ask only about points whose answer is the same a little way off in every direction
(`stable`).
"""

import json
import urllib.request
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
FILES = {"land": "ne_110m_land.geojson", "countries": "ne_110m_admin_0_countries.geojson"}
CACHE = Path.home() / ".cache" / "ojas" / "decision-eval"


def _load(kind):
    path = CACHE / FILES[kind]
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{SOURCE}/{FILES[kind]}", timeout=60) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text())


class Shape:
    """One feature's polygons (outer rings with their holes) and bounding box."""

    def __init__(self, geometry, properties):
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        self.polygons = polygons
        self.properties = properties
        xs = [x for poly in polygons for x, _ in poly[0]]
        ys = [y for poly in polygons for _, y in poly[0]]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))

    def contains(self, lon, lat):
        x0, y0, x1, y1 = self.bbox
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            return False
        for outer, *holes in self.polygons:
            if _inside(outer, lon, lat) and not any(_inside(h, lon, lat) for h in holes):
                return True
        return False


def _inside(ring, x, y):
    """Ray casting: whether (x, y) lies inside the closed ring."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class World:
    def __init__(self):
        self.land = [Shape(f["geometry"], f["properties"]) for f in _load("land")["features"]]
        self.countries = [
            Shape(f["geometry"], f["properties"]) for f in _load("countries")["features"]
            if f["properties"].get("CONTINENT") != "Seven seas (open ocean)"
        ]

    def is_land(self, lat, lon):
        return any(s.contains(lon, lat) for s in self.land)

    def country(self, lat, lon):
        """`(country name, continent)` at the point, or `None` at sea."""
        for s in self.countries:
            if s.contains(lon, lat):
                return s.properties["ADMIN"], s.properties["CONTINENT"]
        return None

    def stable(self, lat, lon, value, margin=1.5):
        """Whether `value(lat, lon)` is unchanged `margin` degrees off in eight directions."""
        here = value(lat, lon)
        for dlat in (-margin, 0.0, margin):
            for dlon in (-margin, 0.0, margin):
                lat2, lon2 = max(-89.9, min(89.9, lat + dlat)), ((lon + dlon + 180.0) % 360.0) - 180.0
                if value(lat2, lon2) != here:
                    return False
        return True
