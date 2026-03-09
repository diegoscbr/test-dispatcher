import csv
import os


DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # --- Build lookup: provider_slug -> aa_slug (only matched rows) ---
    slug_lookup = {}
    with open(os.path.join(DIR, "key.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["aa_slug"]:
                slug_lookup[row["provider_slug"]] = row["aa_slug"]

    # --- Build lookup: aa_slug -> {release_date, intelligence_index} ---
    aa_lookup = {}
    with open(os.path.join(DIR, "consumer_ai_integrations.csv"), newline="") as f:
        for row in csv.DictReader(f):
            aa_lookup[row["slug"]] = {
                "release_date": row["release_date"],
                "intelligence_index": row["eval_artificial_analysis_intelligence_index"],
                "price_1m_blended": row["pricing_price_1m_blended_3_to_1"],
                "price_1m_input": row["pricing_price_1m_input_tokens"],
                "price_1m_output": row["pricing_price_1m_output_tokens"],
                "output_tokens_per_sec": row["median_output_tokens_per_second"],
                "ttft_seconds": row["median_time_to_first_token_seconds"],
                "ttfa_seconds": row["median_time_to_first_answer_token"],
            }

    # --- Read phase2_responses, enrich, and write output ---
    in_path = os.path.join(DIR, "phase2_responses.csv")
    out_path = os.path.join(DIR, "phase2_responses_enriched.csv")

    matched = 0
    total = 0

    with open(in_path, newline="", encoding="utf-8-sig") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        # Deduplicate fieldnames (drop trailing duplicates like p2_response.1)
        seen = set()
        fields = []
        for name in reader.fieldnames:
            if name in seen:
                continue
            seen.add(name)
            fields.append(name)

        # Insert new columns right after 'model'
        model_idx = fields.index("model") + 1
        new_cols = [
            "release_date", "intelligence_index",
            "price_1m_blended", "price_1m_input", "price_1m_output",
            "output_tokens_per_sec", "ttft_seconds", "ttfa_seconds",
        ]
        out_fields = fields[:model_idx] + new_cols + fields[model_idx:]

        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            # Strip duplicate keys from row
            row = {k: row[k] for k in fields}

            # Lookup chain: model -> aa_slug -> AA metadata
            aa_slug = slug_lookup.get(row["model"])
            meta = aa_lookup.get(aa_slug, {}) if aa_slug else {}

            for col in new_cols:
                row[col] = meta.get(col, "")

            if row["intelligence_index"]:
                matched += 1
            total += 1

            writer.writerow(row)

    print(f"Wrote {out_path}")
    print(f"  {total} rows, {matched} matched, {total - matched} unmatched")


if __name__ == "__main__":
    main()
