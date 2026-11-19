"""Tests for the offline OSM-to-road-graph converter."""

from pathlib import Path

import pytest

from core.data.load_road_graph import load_road_graph
from scripts.build_hcmc_osm_graph import Region, build_graph, parse_speed


OSM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="rapo-test">
  <node id="1" lat="10.7000" lon="106.6000" version="1"/>
  <node id="2" lat="10.7000" lon="106.6010" version="1"/>
  <node id="3" lat="10.7000" lon="106.6020" version="1"/>
  <node id="4" lat="10.7010" lon="106.6020" version="1"/>
  <node id="10" lat="10.7100" lon="106.6100" version="1"/>
  <node id="11" lat="10.7110" lon="106.6100" version="1"/>
  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/>
    <tag k="highway" v="primary"/><tag k="name" v="Main Road"/><tag k="maxspeed" v="50"/>
  </way>
  <way id="101" version="1">
    <nd ref="3"/><nd ref="4"/>
    <tag k="highway" v="secondary"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="30 mph"/>
  </way>
  <way id="102" version="1">
    <nd ref="4"/><nd ref="1"/>
    <tag k="highway" v="tertiary"/><tag k="oneway" v="yes"/>
  </way>
  <way id="200" version="1">
    <nd ref="10"/><nd ref="11"/>
    <tag k="highway" v="residential"/>
  </way>
</osm>
"""


def test_parse_speed_handles_metric_mph_and_condensed_values():
    assert parse_speed("50") == 50
    assert parse_speed("30 mph") == pytest.approx(48.28032)
    assert parse_speed("60;40") == 40
    assert parse_speed("signals") is None


def test_build_graph_simplifies_ways_and_drops_isolated_component(tmp_path: Path):
    source = tmp_path / "fixture.osm"
    destination = tmp_path / "road.json"
    source.write_text(OSM_FIXTURE, encoding="utf-8")

    payload = build_graph(
        source,
        destination,
        Region((106.59, 10.69, 106.62, 10.72)),
        component_mode="strong",
        progress_every=0,
    )

    assert {node["id"] for node in payload["nodes"]} == {1, 3, 4}
    assert len(payload["edges"]) == 3
    primary = next(edge for edge in payload["edges"] if edge["osm_way_ids"] == [100])
    assert (primary["from"], primary["to"]) == (1, 3)
    assert primary["bidirectional"] is True
    assert primary["distance_m"] > 200
    secondary = next(edge for edge in payload["edges"] if edge["osm_way_ids"] == [101])
    assert (secondary["from"], secondary["to"]) == (3, 4)
    assert secondary["bidirectional"] is False
    assert secondary["free_flow_speed_kmh"] == pytest.approx(48.28, abs=0.01)
    assert payload["stats"]["removed_component_nodes"] == 2

    nodes, edges = load_road_graph(destination)
    assert set(nodes) == {1, 3, 4}
    assert len(edges) == 3
