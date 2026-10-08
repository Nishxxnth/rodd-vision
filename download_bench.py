"""
download_bench.py – Download freely-licensed benchmark images for bench_check.py
Uses Wikimedia Commons direct URLs and Unsplash Source (no-auth required).
Creates bench_images/<class>/ and saves 3-5 images per class.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).parent / "bench_images"

# fmt: (class_name, [(url, filename), ...])
# All images are freely-licensed (CC0, CC-BY, Unsplash free-to-use)
BENCH_IMAGES = {
    "stair": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Steps_at_Canary_Wharf_DLR_station.jpg/1280px-Steps_at_Canary_Wharf_DLR_station.jpg", "stair_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Stairway_to_Heaven_%28Oahu%2C_Hawaii%29_-_panoramio.jpg/1280px-Stairway_to_Heaven_%28Oahu%2C_Hawaii%29_-_panoramio.jpg", "stair_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Stairs_in_Faroe_Islands.jpg/1280px-Stairs_in_Faroe_Islands.jpg", "stair_03.jpg"),
    ],
    "door": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Red_door.jpg/800px-Red_door.jpg", "door_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Altes_Tor_Neubrandenburg.jpg/800px-Altes_Tor_Neubrandenburg.jpg", "door_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Brown_wooden_door.jpg/800px-Brown_wooden_door.jpg", "door_03.jpg"),
    ],
    "pit": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Pothole_in_road.jpg/800px-Pothole_in_road.jpg", "pit_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Pothole_on_Mtwapa_Road%2C_Kenya.jpg/1280px-Pothole_on_Mtwapa_Road%2C_Kenya.jpg", "pit_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Pothole_Damage.jpg/800px-Pothole_Damage.jpg", "pit_03.jpg"),
    ],
    "electric pole": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Power_pole.jpg/800px-Power_pole.jpg", "epole_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Utility_pole_in_the_US.jpg/800px-Utility_pole_in_the_US.jpg", "epole_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Electric_pole_on_roadside.jpg/800px-Electric_pole_on_roadside.jpg", "epole_03.jpg"),
    ],
    "pillar": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Pillars_of_the_Parthenon.jpg/1280px-Pillars_of_the_Parthenon.jpg", "pillar_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Columns_of_a_Temple.jpg/800px-Columns_of_a_Temple.jpg", "pillar_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Concrete_pillar.jpg/800px-Concrete_pillar.jpg", "pillar_03.jpg"),
    ],
    "dustbin": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Trash_can_-_Stierch.jpg/800px-Trash_can_-_Stierch.jpg", "dustbin_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Outdoor_trash_can.jpg/800px-Outdoor_trash_can.jpg", "dustbin_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Rubbish_bin.jpg/800px-Rubbish_bin.jpg", "dustbin_03.jpg"),
    ],
    "tree": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1280px-24701-nature-natural-beauty.jpg", "tree_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/RedwoodNationalPark-Hiker.jpg/1280px-RedwoodNationalPark-Hiker.jpg", "tree_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Sylhet_tea_garden.jpg/1280px-Sylhet_tea_garden.jpg", "tree_03.jpg"),
    ],
    "street chair": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Park-bench-with-people.jpg/1280px-Park-bench-with-people.jpg", "bench_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Park_bench_in_Berlin.jpg/1280px-Park_bench_in_Berlin.jpg", "bench_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Park_bench_%28Unsplash%29.jpg/1280px-Park_bench_%28Unsplash%29.jpg", "bench_03.jpg"),
    ],
    "road sign": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Road_signs_in_USA.jpg/800px-Road_signs_in_USA.jpg", "roadsign_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Road_signs_in_India.jpg/800px-Road_signs_in_India.jpg", "roadsign_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Stop_sign_MUTCD.svg/800px-Stop_sign_MUTCD.svg.png", "roadsign_03.jpg"),
    ],
    "tram": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Tram_in_Amsterdam.jpg/1280px-Tram_in_Amsterdam.jpg", "tram_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Prague_tram.jpg/1280px-Prague_tram.jpg", "tram_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Melbourne_tram.jpg/1280px-Melbourne_tram.jpg", "tram_03.jpg"),
    ],
    "building": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/800px-Empire_State_Building_%28aerial_view%29.jpg", "building_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/800px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg", "building_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Shopping_Center_in_Shenzhen.jpg/1280px-Shopping_Center_in_Shenzhen.jpg", "building_03.jpg"),
    ],
    # Indoor classes
    "chair": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/800px-Above_Gotham.jpg", "chair_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Rocking_chair_at_Grand_Canyon_Lodge.jpg/800px-Rocking_chair_at_Grand_Canyon_Lodge.jpg", "chair_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Chair_on_white_background.jpg/800px-Chair_on_white_background.jpg", "chair_03.jpg"),
    ],
    "table": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Dining_table_at_Audley_End.jpg/1280px-Dining_table_at_Audley_End.jpg", "table_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/White_shark.jpg/800px-White_shark.jpg", "table_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Huckleberry_Finn_pg_11.jpg/800px-Huckleberry_Finn_pg_11.jpg", "table_03.jpg"),
    ],
    "person": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg", "person_01.jpg"),
        ("https://ultralytics.com/images/zidane.jpg", "person_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg", "person_03.jpg"),
    ],
    "bus": [
        ("https://ultralytics.com/images/bus.jpg", "bus_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/London_Buses_route_149_Walthamstow.jpg/1280px-London_Buses_route_149_Walthamstow.jpg", "bus_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/NYC_Bus.jpg/1280px-NYC_Bus.jpg", "bus_03.jpg"),
    ],
    "car": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2021_BMW_M4_Competition_in_Portimao_Blue%2C_front_8.15.21.jpg/1280px-2021_BMW_M4_Competition_in_Portimao_Blue%2C_front_8.15.21.jpg", "car_01.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/800px-Dog_Breeds.jpg", "car_02.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/2017_Mazda_CX-5_%28KF%29_Maxx_2WD_wagon_%282018-11-02%29_01.jpg/1280px-2017_Mazda_CX-5_%28KF%29_Maxx_2WD_wagon_%282018-11-02%29_01.jpg", "car_03.jpg"),
    ],
}


def download(url: str, dest: Path, retries: int = 2) -> bool:
    """Download url to dest. Returns True on success."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 rodd-bench-downloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if len(data) < 1000:
                print(f"    [WARN] too small ({len(data)} bytes), skipping")
                return False
            dest.write_bytes(data)
            return True
        except Exception as exc:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"    [FAIL] {exc}")
    return False


def main() -> None:
    print(f"Downloading benchmark images to {BENCH_DIR}/")
    ok_total = 0
    fail_total = 0
    for cls_name, images in BENCH_IMAGES.items():
        cls_dir = BENCH_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for url, fname in images:
            dest = cls_dir / fname
            if dest.exists() and dest.stat().st_size > 1000:
                print(f"  [SKIP] {cls_name}/{fname}  (already exists)")
                ok_total += 1
                continue
            print(f"  [DOWN] {cls_name}/{fname} …", end="", flush=True)
            ok = download(url, dest)
            if ok:
                print(f" OK ({dest.stat().st_size // 1024} kB)")
                ok_total += 1
            else:
                fail_total += 1
    print(f"\nDone. {ok_total} downloaded, {fail_total} failed.")
    if fail_total > 0:
        print("  Some downloads failed. Add images manually to bench_images/<class>/")


if __name__ == "__main__":
    main()
