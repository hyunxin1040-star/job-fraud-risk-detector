from functools import lru_cache
from itertools import permutations
from math import atan2, cos, factorial, radians, sin, sqrt
import re

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# 접근성 경로 평가에 필요한 OSM 태그를 보존합니다.
EXTRA_OSM_TAGS = [
    "wheelchair",
    "surface",
    "smoothness",
    "incline",
    "kerb",
    "kerb:height",
    "width",
    "highway",
    "footway",
]
ox.settings.useful_tags_way = list(
    dict.fromkeys(list(ox.settings.useful_tags_way) + EXTRA_OSM_TAGS)
)


st.set_page_config(
    page_title="배프맵",
    page_icon="♿",
    layout="wide",
)

st.title("♿ 배프맵")
st.write("필요한 편의시설을 기준으로 장소를 고르고 이동 경로를 비교해보세요.")


# --------------------------------------------------
# 1. 공통 보조 함수
# --------------------------------------------------
def yes_no(value):
    return "있음" if value else "없음"


def first_value(value):
    """OSM 태그가 리스트인 경우 첫 번째 값을 사용합니다."""
    if isinstance(value, (list, tuple, set)):
        return next(iter(value), None)
    return value


def normalized_text(value):
    value = first_value(value)
    if value is None:
        return ""
    return str(value).strip().lower()


def calculate_distance(lat1, lon1, lat2, lon2):
    """두 위도·경도 사이의 직선거리를 km 단위로 계산합니다."""
    earth_radius = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    latitude_difference = lat2 - lat1
    longitude_difference = lon2 - lon1

    value = (
        sin(latitude_difference / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(longitude_difference / 2) ** 2
    )

    central_angle = 2 * atan2(sqrt(value), sqrt(1 - value))
    return earth_radius * central_angle


def parse_incline_percent(value):
    """
    incline 값을 가능한 경우 퍼센트 숫자로 변환합니다.
    예: '8%' -> 8.0, '-5%' -> 5.0
    up/down/yes 등 수치가 아닌 값은 None으로 처리합니다.
    """
    text = normalized_text(value)

    if not text:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return abs(float(match.group()))
    except ValueError:
        return None


def parse_width_meters(value):
    """width 태그에서 미터 단위 숫자를 가능한 범위에서 추출합니다."""
    text = normalized_text(value)

    if not text:
        return None

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        width = float(match.group())
    except ValueError:
        return None

    # cm 표기가 있으면 m로 변환합니다.
    if "cm" in text:
        width /= 100

    return width


# --------------------------------------------------
# 2. OSM 보행 네트워크 및 접근성 비용
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_walking_graph(center_latitude, center_longitude, radius_meters):
    """OpenStreetMap에서 보행 네트워크를 내려받습니다."""
    graph = ox.graph.graph_from_point(
        center_point=(center_latitude, center_longitude),
        dist=radius_meters,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )

    return add_accessibility_costs(graph)


def edge_accessibility_profile(edge_data):
    """
    길 하나의 접근성 비용과 상태를 계산합니다.

    반환값:
    - multiplier: 거리 비용 배수
    - blocked: 휠체어 경로에서 제외할지 여부
    - warning: 주의 구간인지 여부
    - known_count: 확인 가능한 접근성 태그 수
    - total_count: 확인 대상으로 삼은 태그 수
    - reasons: 판단 이유
    """
    multiplier = 1.0
    blocked = False
    warning = False
    reasons = []

    highway = normalized_text(edge_data.get("highway"))
    wheelchair = normalized_text(edge_data.get("wheelchair"))
    surface = normalized_text(edge_data.get("surface"))
    smoothness = normalized_text(edge_data.get("smoothness"))
    kerb = normalized_text(edge_data.get("kerb"))
    incline = parse_incline_percent(edge_data.get("incline"))
    width = parse_width_meters(edge_data.get("width"))

    checked_values = [
        edge_data.get("wheelchair"),
        edge_data.get("surface"),
        edge_data.get("smoothness"),
        edge_data.get("incline"),
        edge_data.get("kerb"),
        edge_data.get("width"),
    ]
    known_count = sum(value not in (None, "", []) for value in checked_values)
    total_count = len(checked_values)

    # 명백한 차단 조건
    if highway == "steps":
        blocked = True
        reasons.append("계단")

    if wheelchair in {"no", "private"}:
        blocked = True
        reasons.append("휠체어 접근 불가 표시")

    # 긍정 정보
    if wheelchair in {"yes", "designated"}:
        multiplier *= 0.85
        reasons.append("휠체어 접근 가능 표시")

    # 노면
    good_surfaces = {
        "asphalt",
        "concrete",
        "concrete:plates",
        "paving_stones",
        "sett",
    }
    moderate_surfaces = {
        "compacted",
        "fine_gravel",
        "wood",
        "metal",
        "bricks",
    }
    bad_surfaces = {
        "gravel",
        "ground",
        "dirt",
        "earth",
        "sand",
        "grass",
        "mud",
        "pebblestone",
        "rock",
    }

    if surface in good_surfaces:
        multiplier *= 0.95
    elif surface in moderate_surfaces:
        multiplier *= 1.35
        warning = True
        reasons.append(f"주의 노면: {surface}")
    elif surface in bad_surfaces:
        multiplier *= 2.5
        warning = True
        reasons.append(f"불리한 노면: {surface}")
    elif not surface:
        multiplier *= 1.10

    # 평탄도
    if smoothness in {"excellent", "good"}:
        multiplier *= 0.95
    elif smoothness in {"intermediate"}:
        multiplier *= 1.20
        warning = True
        reasons.append("보통 수준의 평탄도")
    elif smoothness in {
        "bad",
        "very_bad",
        "horrible",
        "very_horrible",
        "impassable",
    }:
        multiplier *= 2.5
        warning = True
        reasons.append(f"불량 평탄도: {smoothness}")
    elif not smoothness:
        multiplier *= 1.08

    # 경사도
    if incline is not None:
        if incline > 12:
            multiplier *= 4.0
            warning = True
            reasons.append(f"매우 급한 경사: {incline:g}%")
        elif incline > 8:
            multiplier *= 2.5
            warning = True
            reasons.append(f"급한 경사: {incline:g}%")
        elif incline > 5:
            multiplier *= 1.6
            warning = True
            reasons.append(f"경사 주의: {incline:g}%")
    else:
        multiplier *= 1.05

    # 턱
    if kerb in {"raised", "yes", "regular", "high"}:
        multiplier *= 2.0
        warning = True
        reasons.append(f"턱 주의: {kerb}")
    elif kerb in {"lowered", "flush", "no"}:
        multiplier *= 0.95
    elif not kerb:
        multiplier *= 1.05

    # 폭
    if width is not None:
        if width < 0.9:
            multiplier *= 3.0
            warning = True
            reasons.append(f"통행 폭 협소: {width:g}m")
        elif width < 1.2:
            multiplier *= 1.6
            warning = True
            reasons.append(f"통행 폭 주의: {width:g}m")
    else:
        multiplier *= 1.03

    return {
        "multiplier": multiplier,
        "blocked": blocked,
        "warning": warning,
        "known_count": known_count,
        "total_count": total_count,
        "reasons": reasons,
    }


def add_accessibility_costs(graph):
    """각 edge에 wheelchair_cost와 접근성 진단 정보를 추가합니다."""
    for _, _, _, edge_data in graph.edges(keys=True, data=True):
        length = float(edge_data.get("length", 1.0))
        profile = edge_accessibility_profile(edge_data)

        edge_data["wheelchair_blocked"] = profile["blocked"]
        edge_data["accessibility_warning"] = profile["warning"]
        edge_data["accessibility_known_count"] = profile["known_count"]
        edge_data["accessibility_total_count"] = profile["total_count"]
        edge_data["accessibility_reasons"] = profile["reasons"]

        if profile["blocked"]:
            # 완전 제거 대신 매우 큰 비용을 부여하고 아래에서 경로 후보에서 제외합니다.
            edge_data["wheelchair_cost"] = length * 1_000_000
        else:
            edge_data["wheelchair_cost"] = length * profile["multiplier"]

    return graph


def graph_for_mode(graph, route_mode):
    """
    경로 모드에 맞는 그래프와 최적화 weight를 반환합니다.

    휠체어 친화 모드에서는 명백한 접근 불가 구간만 완전히 제거합니다.
    - highway=steps
    - wheelchair=no/private

    접근성 태그가 없는 길은 제거하지 않고 비용만 높입니다.
    """
    if route_mode == "휠체어 친화 경로":
        wheelchair_graph = graph.copy()

        blocked_edges = [
            (u, v, key)
            for u, v, key, data in wheelchair_graph.edges(
                keys=True,
                data=True,
            )
            if data.get("wheelchair_blocked", False)
        ]
        wheelchair_graph.remove_edges_from(blocked_edges)

        isolated_nodes = list(nx.isolates(wheelchair_graph))
        wheelchair_graph.remove_nodes_from(isolated_nodes)

        return wheelchair_graph, "wheelchair_cost"

    return graph, "length"


def get_candidate_nodes(graph, place, candidate_count=10):
    """
    장소 주변의 가까운 접근 가능 노드를 반환합니다.
    속도를 위해 기본 후보 수는 3개로 제한합니다.
    """
    if graph.number_of_nodes() == 0:
        return []

    node_ids = []
    latitudes = []
    longitudes = []

    for node_id, node_data in graph.nodes(data=True):
        if "x" not in node_data or "y" not in node_data:
            continue

        node_ids.append(node_id)
        longitudes.append(float(node_data["x"]))
        latitudes.append(float(node_data["y"]))

    if not node_ids:
        return []

    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    target_latitude = float(place["latitude"])
    target_longitude = float(place["longitude"])

    latitude_distance = (latitudes - target_latitude) * 111_000
    longitude_distance = (longitudes - target_longitude) * 88_000
    squared_distance = latitude_distance ** 2 + longitude_distance ** 2

    count = min(candidate_count, len(node_ids))
    nearest_indexes = np.argpartition(
        squared_distance,
        count - 1,
    )[:count]

    sorted_indexes = nearest_indexes[
        np.argsort(squared_distance[nearest_indexes])
    ]

    return [node_ids[index] for index in sorted_indexes]


def find_best_candidate_route(
    graph,
    origin_candidates,
    destination_candidates,
    weight,
):
    """
    여러 출발 후보와 도착 후보를 multi-source Dijkstra 한 번으로 비교합니다.
    후보 조합마다 shortest_path를 반복 호출하는 방식보다 훨씬 빠릅니다.
    """
    super_source = "__temporary_super_source__"
    while super_source in graph:
        super_source += "_"

    working_graph = graph.copy()
    working_graph.add_node(super_source)

    for origin_node in origin_candidates:
        working_graph.add_edge(
            super_source,
            origin_node,
            length=0.0,
            wheelchair_cost=0.0,
        )

    try:
        distances, paths = nx.single_source_dijkstra(
            working_graph,
            source=super_source,
            weight=weight,
        )
    except nx.NetworkXNoPath:
        return None, None, None

    reachable_destinations = [
        node
        for node in destination_candidates
        if node in distances
    ]

    if not reachable_destinations:
        return None, None, None

    best_destination = min(
        reachable_destinations,
        key=lambda node: distances[node],
    )

    route = paths[best_destination][1:]

    if len(route) < 2:
        return None, None, None

    route_edges = ox.routing.route_to_gdf(
        graph,
        route,
        weight=weight,
    )

    real_distance = float(route_edges["length"].sum())
    optimized_cost = float(route_edges[weight].sum())

    return route, real_distance, optimized_cost


def get_network_route(
    graph,
    origin_place,
    destination_place,
    weight,
    candidate_count=10,
):
    """
    장소 주변 후보 노드를 찾고 multi-source Dijkstra로 최적 연결을 구합니다.
    """
    origin_candidates = get_candidate_nodes(
        graph,
        origin_place,
        candidate_count=candidate_count,
    )
    destination_candidates = get_candidate_nodes(
        graph,
        destination_place,
        candidate_count=candidate_count,
    )

    if not origin_candidates or not destination_candidates:
        raise nx.NetworkXNoPath(
            "장소 주변에서 접근 가능한 보행 노드를 찾지 못했습니다."
        )

    route, real_distance, optimized_cost = find_best_candidate_route(
        graph,
        origin_candidates,
        destination_candidates,
        weight,
    )

    if route is None:
        raise nx.NetworkXNoPath(
            f"{origin_place['name']}에서 {destination_place['name']}까지 "
            "접근 가능한 우회 경로를 찾지 못했습니다."
        )

    return route, real_distance, optimized_cost


def get_or_calculate_segment(
    graph,
    origin_place,
    destination_place,
    weight,
    segment_cache,
    candidate_count=10,
):
    """
    필요한 방향의 구간만 계산하고 캐시에 저장합니다.

    동일한 방향의 구간이 다른 방문 순서에서도 다시 필요하면
    경로를 재계산하지 않고 저장된 결과를 사용합니다.
    """
    pair_key = (
        origin_place["name"],
        destination_place["name"],
    )

    if pair_key in segment_cache:
        return segment_cache[pair_key]

    route, real_distance, optimized_cost = get_network_route(
        graph,
        origin_place,
        destination_place,
        weight,
        candidate_count=candidate_count,
    )

    segment_cache[pair_key] = {
        "route": route,
        "real_distance": real_distance,
        "optimized_cost": optimized_cost,
    }

    return segment_cache[pair_key]


def evaluate_place_order(
    graph,
    place_order,
    weight,
    segment_cache,
    candidate_count=10,
    progress_callback=None,
):
    """
    하나의 방문 순서에 실제로 필요한 방향의 구간만 계산합니다.
    """
    total_real_distance = 0.0
    total_optimized_cost = 0.0
    route_segments = []

    for index in range(len(place_order) - 1):
        origin_place = place_order[index]
        destination_place = place_order[index + 1]

        if progress_callback:
            progress_callback(
                origin_place["name"],
                destination_place["name"],
            )

        segment = get_or_calculate_segment(
            graph,
            origin_place,
            destination_place,
            weight,
            segment_cache,
            candidate_count=candidate_count,
        )

        route_segments.append(segment["route"])
        total_real_distance += segment["real_distance"]
        total_optimized_cost += segment["optimized_cost"]

    return {
        "places": place_order,
        "segments": route_segments,
        "real_distance": total_real_distance,
        "optimized_cost": total_optimized_cost,
    }


def find_best_route_lazy(
    graph,
    start_place,
    destination_places,
    weight,
    candidate_count=10,
    progress_callback=None,
):
    """
    출발지를 고정하고 나머지 장소의 모든 순열을 정확히 비교합니다.

    전체 장소가 n개라면 후보 방문 순서는 정확히 (n-1)!개입니다.
    각 후보 순서에서 필요한 방향의 구간만 계산하며,
    이미 계산한 동일 방향 구간은 캐시에서 재사용합니다.
    """
    segment_cache = {}
    destination_count = len(destination_places)
    total_orders = factorial(destination_count)

    # permutations 객체를 리스트로 만들지 않아 메모리를 아낍니다.
    candidate_orders = permutations(destination_places)

    best_result = None

    for order_index, destination_order in enumerate(
        candidate_orders,
        start=1,
    ):
        place_order = [start_place] + list(destination_order)

        if progress_callback:
            progress_callback(
                None,
                None,
                order_index,
                total_orders,
            )

        try:
            result = evaluate_place_order(
                graph,
                place_order,
                weight,
                segment_cache,
                candidate_count=candidate_count,
                progress_callback=(
                    lambda origin_name, destination_name:
                    progress_callback(
                        origin_name,
                        destination_name,
                        order_index,
                        total_orders,
                    )
                    if progress_callback
                    else None
                ),
            )
        except nx.NetworkXNoPath:
            continue

        if (
            best_result is None
            or result["optimized_cost"] < best_result["optimized_cost"]
        ):
            best_result = result

    optimization_method = (
        f"출발지 고정 정확 최적화 "
        f"({total_orders:,}개 방문 순서 비교)"
    )

    if best_result is None:
        return (
            None,
            None,
            None,
            None,
            optimization_method,
            segment_cache,
        )

    return (
        best_result["places"],
        best_result["segments"],
        best_result["real_distance"],
        best_result["optimized_cost"],
        optimization_method,
        segment_cache,
    )


def route_cost_from_places(place_order, pair_costs):
    """주어진 장소 순서의 총 최적화 비용을 계산합니다."""
    total_cost = 0.0

    for index in range(len(place_order) - 1):
        pair_key = (
            place_order[index]["name"],
            place_order[index + 1]["name"],
        )

        if pair_key not in pair_costs:
            return float("inf")

        total_cost += pair_costs[pair_key]

    return total_cost


def nearest_neighbor_order(start_place, destination_places, pair_costs):
    """
    출발지에서 시작해 아직 방문하지 않은 장소 중
    현재 위치와 비용이 가장 낮은 장소를 차례로 선택합니다.
    """
    route = [start_place]
    remaining = destination_places.copy()

    while remaining:
        current_name = route[-1]["name"]

        reachable_places = [
            place
            for place in remaining
            if (current_name, place["name"]) in pair_costs
        ]

        if not reachable_places:
            return None

        next_place = min(
            reachable_places,
            key=lambda place: pair_costs[
                (current_name, place["name"])
            ],
        )

        route.append(next_place)
        remaining.remove(next_place)

    return route


def improve_route_with_two_opt(place_order, pair_costs):
    """
    출발지는 고정한 채 2-opt 방식으로 방문 순서를 반복 개선합니다.
    열린 경로이므로 마지막 장소에서 출발지로 돌아오지는 않습니다.
    """
    best_order = place_order.copy()
    best_cost = route_cost_from_places(best_order, pair_costs)

    improved = True

    while improved:
        improved = False

        # 인덱스 0은 출발지이므로 뒤집지 않습니다.
        for start_index in range(1, len(best_order) - 1):
            for end_index in range(start_index + 1, len(best_order)):
                candidate_order = (
                    best_order[:start_index]
                    + list(reversed(best_order[start_index:end_index + 1]))
                    + best_order[end_index + 1:]
                )

                candidate_cost = route_cost_from_places(
                    candidate_order,
                    pair_costs,
                )

                if candidate_cost + 1e-9 < best_cost:
                    best_order = candidate_order
                    best_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return best_order, best_cost


def exact_route_order(start_place, destination_places, pair_costs):
    """장소 수가 적을 때 모든 방문 순서를 비교해 정확한 최적해를 찾습니다."""
    best_order = None
    best_cost = float("inf")

    for destination_order in permutations(destination_places):
        candidate_order = [start_place] + list(destination_order)
        candidate_cost = route_cost_from_places(
            candidate_order,
            pair_costs,
        )

        if candidate_cost < best_cost:
            best_order = candidate_order
            best_cost = candidate_cost

    return best_order, best_cost


def find_best_network_route(
    start_place,
    destination_places,
    pair_routes,
    pair_real_distances,
    pair_costs,
):
    """
    선택 장소 수에 따라 최적화 방법을 자동으로 선택합니다.

    - 전체 장소가 8곳 이하: 모든 순서를 비교해 정확한 최적해 산출
    - 9곳 이상: 최근접 이웃으로 초기 경로 생성 후 2-opt로 개선
    """
    total_place_count = len(destination_places) + 1

    if total_place_count <= 8:
        best_places, lowest_cost = exact_route_order(
            start_place,
            destination_places,
            pair_costs,
        )
        optimization_method = "정확 최적화"
    else:
        initial_order = nearest_neighbor_order(
            start_place,
            destination_places,
            pair_costs,
        )

        if initial_order is None:
            return None, None, None, None, None

        best_places, lowest_cost = improve_route_with_two_opt(
            initial_order,
            pair_costs,
        )
        optimization_method = "근사 최적화(최근접 이웃 + 2-opt)"

    if best_places is None:
        return None, None, None, None, None

    best_segments = []
    best_real_distance = 0.0

    for index in range(len(best_places) - 1):
        pair_key = (
            best_places[index]["name"],
            best_places[index + 1]["name"],
        )

        if (
            pair_key not in pair_routes
            or pair_key not in pair_real_distances
        ):
            return None, None, None, None, None

        best_segments.append(pair_routes[pair_key])
        best_real_distance += pair_real_distances[pair_key]

    return (
        best_places,
        best_segments,
        best_real_distance,
        lowest_cost,
        optimization_method,
    )


def get_route_edge(graph, start_node, end_node, weight):
    """평행 edge 중 현재 weight가 가장 작은 edge를 반환합니다."""
    edge_options = graph.get_edge_data(start_node, end_node)

    if not edge_options:
        return None

    return min(
        edge_options.values(),
        key=lambda edge: edge.get(weight, float("inf")),
    )


def route_to_coordinates(graph, route, weight):
    coordinates = []

    for start_node, end_node in zip(route[:-1], route[1:]):
        edge = get_route_edge(graph, start_node, end_node, weight)

        if edge is None:
            continue

        geometry = edge.get("geometry")

        if geometry is not None:
            edge_coordinates = [
                [latitude, longitude]
                for longitude, latitude in geometry.coords
            ]
        else:
            edge_coordinates = [
                [
                    graph.nodes[start_node]["y"],
                    graph.nodes[start_node]["x"],
                ],
                [
                    graph.nodes[end_node]["y"],
                    graph.nodes[end_node]["x"],
                ],
            ]

        if coordinates and edge_coordinates:
            if coordinates[-1] == edge_coordinates[0]:
                edge_coordinates = edge_coordinates[1:]

        coordinates.extend(edge_coordinates)

    return coordinates


def analyze_selected_route(graph, route_segments, weight):
    """추천 경로의 접근성 정보 확인률과 주의 구간을 계산합니다."""
    traversed_edges = 0
    warning_edges = 0
    known_count = 0
    total_count = 0
    warning_details = []

    for route in route_segments:
        for start_node, end_node in zip(route[:-1], route[1:]):
            edge = get_route_edge(graph, start_node, end_node, weight)

            if edge is None:
                continue

            traversed_edges += 1
            known_count += int(edge.get("accessibility_known_count", 0))
            total_count += int(edge.get("accessibility_total_count", 0))

            if edge.get("accessibility_warning", False):
                warning_edges += 1

                for reason in edge.get("accessibility_reasons", []):
                    if reason not in warning_details:
                        warning_details.append(reason)

    information_rate = (
        known_count / total_count * 100
        if total_count
        else 0
    )

    return {
        "edge_count": traversed_edges,
        "warning_count": warning_edges,
        "information_rate": information_rate,
        "warning_details": warning_details,
    }


def calculate_graph_area(selected_places):
    center_latitude = float(selected_places["latitude"].mean())
    center_longitude = float(selected_places["longitude"].mean())

    maximum_distance_km = 0

    for _, place in selected_places.iterrows():
        distance_km = calculate_distance(
            center_latitude,
            center_longitude,
            place["latitude"],
            place["longitude"],
        )
        maximum_distance_km = max(maximum_distance_km, distance_km)

    radius_meters = int(
        max(3000, maximum_distance_km * 1000 + 2000)
    )

    return center_latitude, center_longitude, radius_meters


# --------------------------------------------------
# 3. 임시 장소 데이터
# --------------------------------------------------
places = pd.DataFrame(
    [
        {
            "name": "경복궁",
            "category": "관광지",
            "address": "서울특별시 종로구 사직로 161",
            "latitude": 37.5796,
            "longitude": 126.9770,
            "wheelchair": True,
            "elevator": False,
            "toilet": True,
            "parking": True,
            "ramp": True,
        },
        {
            "name": "서울역",
            "category": "교통시설",
            "address": "서울특별시 중구 한강대로 405",
            "latitude": 37.5547,
            "longitude": 126.9707,
            "wheelchair": True,
            "elevator": True,
            "toilet": True,
            "parking": True,
            "ramp": True,
        },
        {
            "name": "국립중앙박물관",
            "category": "관광지",
            "address": "서울특별시 용산구 서빙고로 137",
            "latitude": 37.5239,
            "longitude": 126.9803,
            "wheelchair": True,
            "elevator": True,
            "toilet": True,
            "parking": True,
            "ramp": True,
        },
        {
            "name": "서울시립미술관",
            "category": "관광지",
            "address": "서울특별시 중구 덕수궁길 61",
            "latitude": 37.5641,
            "longitude": 126.9738,
            "wheelchair": True,
            "elevator": True,
            "toilet": True,
            "parking": False,
            "ramp": True,
        },
        {
            "name": "북촌한옥마을",
            "category": "관광지",
            "address": "서울특별시 종로구 계동길 37",
            "latitude": 37.5826,
            "longitude": 126.9831,
            "wheelchair": False,
            "elevator": False,
            "toilet": False,
            "parking": False,
            "ramp": False,
        },
    ]
)


# --------------------------------------------------
# 4. 사이드바 검색 및 필터
# --------------------------------------------------
st.sidebar.header("검색 조건")

search_keyword = st.sidebar.text_input(
    "장소 이름 검색",
    placeholder="예: 서울역",
)

category_options = sorted(places["category"].unique())

selected_categories = st.sidebar.multiselect(
    "장소 유형",
    options=category_options,
    default=category_options,
)

st.sidebar.subheader("필수 편의시설")

need_elevator = st.sidebar.checkbox("엘리베이터")
need_toilet = st.sidebar.checkbox("장애인 화장실")
need_parking = st.sidebar.checkbox("장애인 주차장")
need_ramp = st.sidebar.checkbox("경사로")


# --------------------------------------------------
# 5. 필터링
# --------------------------------------------------
filtered_places = places.copy()

if search_keyword:
    filtered_places = filtered_places[
        filtered_places["name"].str.contains(
            search_keyword,
            case=False,
            na=False,
        )
    ]

filtered_places = filtered_places[
    filtered_places["category"].isin(selected_categories)
]

if need_elevator:
    filtered_places = filtered_places[filtered_places["elevator"]]

if need_toilet:
    filtered_places = filtered_places[filtered_places["toilet"]]

if need_parking:
    filtered_places = filtered_places[filtered_places["parking"]]

if need_ramp:
    filtered_places = filtered_places[filtered_places["ramp"]]


# --------------------------------------------------
# 6. 검색 결과
# --------------------------------------------------
st.subheader("검색 결과")
st.metric("조건에 맞는 장소", f"{len(filtered_places)}곳")

if filtered_places.empty:
    st.warning("조건에 맞는 장소가 없습니다. 필터 조건을 줄여보세요.")
    st.stop()


# --------------------------------------------------
# 7. 여행 경로 계산
# --------------------------------------------------
st.subheader("여행 경로 만들기")
st.caption(
    "방문할 장소와 출발지를 선택하면 가장 적합한 이동 순서를 추천해드려요."
)

route_mode = st.radio(
    "경로 기준",
    options=[
        "가장 짧은 보행 경로",
        "휠체어 친화 경로",
    ],
    horizontal=True,
    help=(
        "휠체어 친화 경로는 계단과 wheelchair=no 구간을 완전히 제외하고, "
        "계단과 휠체어 접근이 어려운 구간을 피해서 이동 가능한 길을 찾아요."
    ),
)

available_place_names = filtered_places["name"].tolist()

selected_place_names = st.multiselect(
    "방문할 장소를 2곳 이상 선택하세요",
    options=available_place_names,
    placeholder="장소를 선택하세요",
)

MAX_EXACT_PLACES = 10

if len(selected_place_names) > MAX_EXACT_PLACES:
    st.warning(
        f"한 번에 최대 {MAX_EXACT_PLACES}곳까지 선택할 수 있어요."
    )

elif len(selected_place_names) >= 2:
    start_place_name = st.selectbox(
        "출발 장소",
        options=selected_place_names,
    )

    route_candidate_count = factorial(
        len(selected_place_names) - 1
    )

    if len(selected_place_names) >= 8:
        st.info(
            "선택한 장소가 많아 경로 추천에 시간이 조금 걸릴 수 있어요."
        )

    if st.button("최적 경로 추천", type="primary"):
        selected_places = filtered_places[
            filtered_places["name"].isin(selected_place_names)
        ].copy()

        selected_records = selected_places.to_dict("records")

        start_place = next(
            place
            for place in selected_records
            if place["name"] == start_place_name
        )

        destination_places = [
            place
            for place in selected_records
            if place["name"] != start_place_name
        ]

        (
            center_latitude,
            center_longitude,
            radius_meters,
        ) = calculate_graph_area(selected_places)

        try:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            with st.spinner(
                "가장 적합한 이동 경로를 찾고 있어요."
            ):
                best_route = None
                best_route_segments = None
                shortest_distance = None
                optimized_cost = None
                optimization_method = None
                routing_graph = None
                weight = None
                used_radius_meters = None
                calculated_segment_count = 0

                # 필요한 구간만 계산하므로 우회 탐색 범위를 다시 3단계로 넓힙니다.
                radius_multipliers = [1.0, 1.7, 2.5]

                for stage_index, radius_multiplier in enumerate(
                    radius_multipliers,
                    start=1,
                ):
                    current_radius = int(
                        radius_meters * radius_multiplier
                    )

                    progress_text.write(
                        "주변 이동 경로를 확인하고 있어요."
                    )

                    base_graph = load_walking_graph(
                        round(center_latitude, 5),
                        round(center_longitude, 5),
                        current_radius,
                    )

                    current_graph, current_weight = graph_for_mode(
                        base_graph,
                        route_mode,
                    )

                    def update_progress(
                        origin_name,
                        destination_name,
                        order_index,
                        total_orders,
                    ):
                        stage_fraction = (
                            (stage_index - 1)
                            + order_index / max(total_orders, 1)
                        ) / len(radius_multipliers)

                        progress_bar.progress(
                            min(stage_fraction, 1.0)
                        )

                        if origin_name and destination_name:
                            progress_text.write(
                                f"{origin_name}에서 {destination_name}까지 확인 중"
                            )
                        else:
                            progress_text.write(
                                "방문 순서를 비교하고 있어요."
                            )

                    try:
                        (
                            candidate_best_route,
                            candidate_best_segments,
                            candidate_distance,
                            candidate_cost,
                            candidate_method,
                            segment_cache,
                        ) = find_best_route_lazy(
                            current_graph,
                            start_place,
                            destination_places,
                            current_weight,
                            candidate_count=10,
                            progress_callback=update_progress,
                        )

                    except nx.NetworkXNoPath:
                        continue

                    if candidate_best_route is not None:
                        best_route = candidate_best_route
                        best_route_segments = candidate_best_segments
                        shortest_distance = candidate_distance
                        optimized_cost = candidate_cost
                        optimization_method = candidate_method
                        routing_graph = current_graph
                        weight = current_weight
                        used_radius_meters = current_radius
                        calculated_segment_count = len(segment_cache)
                        break

                progress_bar.progress(1.0)
                progress_text.empty()

            if best_route is None:
                st.error(
                    "검색 범위를 세 차례 넓히고 장소 주변의 접근 가능한 "
                    "진입 노드도 여러 개 비교했지만 경로를 찾지 못했습니다. "
                    "이 경우 해당 장소 좌표가 실제 접근 가능한 출입구와 "
                    "다를 가능성이 큽니다."
                )
            else:
                route_analysis = analyze_selected_route(
                    routing_graph,
                    best_route_segments,
                    weight,
                )

                st.session_state["routing_graph"] = routing_graph
                st.session_state["routing_weight"] = weight
                st.session_state["route_mode"] = route_mode
                st.session_state["best_route"] = best_route
                st.session_state["best_route_segments"] = best_route_segments
                st.session_state["shortest_distance"] = shortest_distance
                st.session_state["optimized_cost"] = optimized_cost
                st.session_state["optimization_method"] = optimization_method
                st.session_state["route_analysis"] = route_analysis
                st.session_state["used_radius_meters"] = used_radius_meters
                st.session_state["calculated_segment_count"] = (
                    calculated_segment_count
                )

        except nx.NetworkXNoPath:
            st.error(
                "선택한 장소 사이에 연결 가능한 경로가 없습니다. "
                "휠체어 친화 모드에서는 계단과 접근 불가 구간을 제외하므로 "
                "경로가 끊길 수 있습니다."
            )
        except Exception as error:
            st.error(
                "경로 계산 중 오류가 발생했습니다. "
                "인터넷 연결을 확인한 뒤 다시 시도해 주세요."
            )
            st.exception(error)

elif len(selected_place_names) == 1:
    st.info("경로를 만들려면 장소를 한 곳 더 선택하세요.")


routing_graph = st.session_state.get("routing_graph")
routing_weight = st.session_state.get("routing_weight")
saved_route_mode = st.session_state.get("route_mode")
best_route = st.session_state.get("best_route")
best_route_segments = st.session_state.get("best_route_segments")
shortest_distance = st.session_state.get("shortest_distance")
route_analysis = st.session_state.get("route_analysis")
optimization_method = st.session_state.get("optimization_method")
used_radius_meters = st.session_state.get("used_radius_meters")
calculated_segment_count = st.session_state.get(
    "calculated_segment_count"
)

if best_route:
    current_names = set(filtered_places["name"])
    route_names_set = {place["name"] for place in best_route}

    if not route_names_set.issubset(current_names):
        for key in [
            "routing_graph",
            "routing_weight",
            "route_mode",
            "best_route",
            "best_route_segments",
            "shortest_distance",
            "optimized_cost",
            "optimization_method",
            "route_analysis",
            "used_radius_meters",
            "calculated_segment_count",
        ]:
            st.session_state.pop(key, None)

        routing_graph = None
        routing_weight = None
        saved_route_mode = None
        best_route = None
        best_route_segments = None
        shortest_distance = None
        route_analysis = None
        optimization_method = None


# --------------------------------------------------
# 8. 지도
# --------------------------------------------------
if best_route:
    map_center = [
        sum(place["latitude"] for place in best_route) / len(best_route),
        sum(place["longitude"] for place in best_route) / len(best_route),
    ]
    zoom_start = 13
else:
    map_center = [
        filtered_places["latitude"].mean(),
        filtered_places["longitude"].mean(),
    ]
    zoom_start = 12

seoul_map = folium.Map(
    location=map_center,
    zoom_start=zoom_start,
)

for _, place in filtered_places.iterrows():
    popup_text = f"""
    <div style="width: 230px;">
        <h4>{place['name']}</h4>
        <b>분류:</b> {place['category']}<br>
        <b>주소:</b> {place['address']}<br><br>
        ♿ 휠체어 접근: {yes_no(place['wheelchair'])}<br>
        🛗 엘리베이터: {yes_no(place['elevator'])}<br>
        🚻 장애인 화장실: {yes_no(place['toilet'])}<br>
        🅿️ 장애인 주차장: {yes_no(place['parking'])}<br>
        ↗️ 경사로: {yes_no(place['ramp'])}
    </div>
    """

    marker_color = "green" if place["wheelchair"] else "red"

    folium.Marker(
        location=[place["latitude"], place["longitude"]],
        tooltip=place["name"],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(
            color=marker_color,
            icon="info-sign",
        ),
    ).add_to(seoul_map)


if (
    routing_graph is not None
    and routing_weight
    and best_route
    and best_route_segments
):
    all_route_coordinates = []

    for route_segment in best_route_segments:
        segment_coordinates = route_to_coordinates(
            routing_graph,
            route_segment,
            routing_weight,
        )

        if segment_coordinates:
            folium.PolyLine(
                locations=segment_coordinates,
                weight=7,
                opacity=0.88,
                tooltip=saved_route_mode or "추천 경로",
            ).add_to(seoul_map)

            all_route_coordinates.extend(segment_coordinates)

    for order, place in enumerate(best_route, start=1):
        folium.Marker(
            location=[place["latitude"], place["longitude"]],
            tooltip=f"{order}. {place['name']}",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background-color: white;
                    border: 3px solid black;
                    border-radius: 50%;
                    width: 34px;
                    height: 34px;
                    text-align: center;
                    line-height: 28px;
                    font-weight: bold;
                    font-size: 16px;
                ">
                    {order}
                </div>
                """
            ),
        ).add_to(seoul_map)

    if all_route_coordinates:
        seoul_map.fit_bounds(all_route_coordinates)


st_folium(
    seoul_map,
    width=None,
    height=620,
)


# --------------------------------------------------
# 9. 경로 결과
# --------------------------------------------------
if best_route and shortest_distance is not None:
    st.subheader("추천 방문 순서")

    st.success(
        " → ".join(place["name"] for place in best_route)
    )

    st.write(f"**추천 기준:** {saved_route_mode}")


    estimated_minutes = round(
        shortest_distance / 1000 / 3 * 60
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric(
        "총 이동거리",
        f"{shortest_distance / 1000:.2f} km",
    )
    metric_col2.metric(
        "예상 이동시간",
        f"약 {estimated_minutes}분",
    )

    if route_analysis:
        metric_col3.metric(
            "주의 구간",
            f"{route_analysis['warning_count']}개",
        )
        metric_col4.metric(
            "접근성 정보 확인률",
            f"{route_analysis['information_rate']:.0f}%",
        )

    st.caption(
        "예상 이동시간은 시속 3km를 가정한 단순 추정치이며 "
        "신호 대기, 휴식, 실제 휠체어 이동 속도는 반영하지 않습니다."
    )

    leg_rows = []

    for index in range(len(best_route) - 1):
        origin_name = best_route[index]["name"]
        destination_name = best_route[index + 1]["name"]

        segment_route = best_route_segments[index]
        segment_edges = ox.routing.route_to_gdf(
            routing_graph,
            segment_route,
            weight=routing_weight,
        )
        segment_distance = float(segment_edges["length"].sum())

        leg_rows.append(
            {
                "순서": index + 1,
                "출발": origin_name,
                "도착": destination_name,
                "거리(km)": round(segment_distance / 1000, 2),
                "예상시간(분)": round(segment_distance / 1000 / 3 * 60),
            }
        )

    with st.expander("구간별 이동 정보"):
        st.dataframe(
            pd.DataFrame(leg_rows),
            use_container_width=True,
            hide_index=True,
        )

    if route_analysis and route_analysis["warning_details"]:
        with st.expander("경로상 확인된 주의 요소"):
            for detail in route_analysis["warning_details"]:
                st.write(f"- {detail}")

    if saved_route_mode == "휠체어 친화 경로":
        st.warning(
            "현장 상황이나 시설 상태가 달라질 수 있으니 이동 전 최신 정보를 확인해 주세요."
        )
    else:
        st.info(
            "가장 짧은 경로는 이동 편의시설보다 거리를 우선해 추천해요."
        )


# --------------------------------------------------
# 10. 장소 상세 정보
# --------------------------------------------------
st.subheader("장소 상세 정보")

for _, place in filtered_places.iterrows():
    with st.expander(f"{place['name']} · {place['category']}"):
        st.write(f"**주소:** {place['address']}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("엘리베이터", yes_no(place["elevator"]))
        col2.metric("장애인 화장실", yes_no(place["toilet"]))
        col3.metric("장애인 주차장", yes_no(place["parking"]))
        col4.metric("경사로", yes_no(place["ramp"]))
