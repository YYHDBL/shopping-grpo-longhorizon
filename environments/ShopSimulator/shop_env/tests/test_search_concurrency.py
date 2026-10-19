from concurrent.futures import ThreadPoolExecutor

from web_agent_site.engine.search import MultiFieldBM25Searcher, build_index


def test_shared_connection_matches_serial_ranking(tmp_path):
    products = [
        {"asin": str(i), "title": ("red cup" if i % 2 else "blue bowl"), "brand": "shop"}
        for i in range(80)
    ]
    path = tmp_path / "products.sqlite3"
    build_index(products, path, product_data_sha256="fixture")
    search = MultiFieldBM25Searcher(path, expected_product_sha256="fixture")
    queries = ["red cup", "blue bowl", "shop", "missing"]
    expected = {q: search.search(q) for q in queries}

    def query(i):
        q = queries[i % len(queries)]
        assert search.contains_asin("1")
        assert search.search(q) == expected[q]

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(query, range(800)))
    finally:
        search.close()
