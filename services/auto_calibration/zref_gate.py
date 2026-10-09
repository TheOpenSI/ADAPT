import json
import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import numpy as np
import pandas as pd


class ZReferenceGate:

    def __init__(self, model_name, services, reference_queries, scorer,
                 cache_dir="calibration_cache", percentile=10, top_n=5, verbose=True):
        """
        model_name         name of the model (used for the cache file name)
        services           dict: specialist name -> description
        reference_queries  list of ~100 off-topic queries (keep the same list for every run)
        scorer             function(model_name, queries, description) -> list of scores
        percentile         10 = about 10% of off-topic queries are allowed through
        top_n              5 = the lead compares the best score with the next 4 scores
        """
        self.model_name = model_name
        self.services = services
        self.names = list(services.keys())
        self.reference_queries = list(reference_queries)
        self.scorer = scorer
        self.percentile = percentile
        self.top_n = top_n
        self.verbose = verbose
        self.cache_note = ""

        os.makedirs(cache_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_")
        self.cache_file = os.path.join(cache_dir, safe_name + ".csv")

        self.calibrate()


    def load_cache(self):
        """
        Return the saved reference scores if they can be reused, otherwise an empty cache.

        Returns a dict: {"reference_queries": [...], "columns": {"name|description": [scores]}}
        """
        empty = {"reference_queries": self.reference_queries, "columns": {}}

        if not os.path.exists(self.cache_file):
            return empty

        with open(self.cache_file) as f:
            cache = json.load(f)

        if cache["reference_queries"] != self.reference_queries:
            return empty
        return cache

    def shortlist_lead(self, scores): 
        
        """Best score minus the average of the next few scores (for ONE query)."""
        
        sorted_scores = sorted(scores, reverse=True) 
        shortlist = sorted_scores[:self.top_n]     
        best = shortlist[0]
        others = shortlist[1:]  
        return best - np.mean(others)    


    def load_table(self):
        """Return the saved score table, or an empty one if there is none or it used other reference queries."""

        # an empty table: one row per reference query, no columns yet
        empty = pd.DataFrame(index=self.reference_queries)
        
        # first run: no saved file, so start from the empty table
        if not os.path.exists(self.cache_file):
            return empty

        # read the saved file (the first column holds the query text)
        table = pd.read_csv(self.cache_file, index_col=0)

        # just checking: if the saved queries are not the same as the current reference queries, start again
        if list(table.index) != self.reference_queries:
            return empty

        return table


    def calibrate(self):
        
        # Step 1: load the saved table (or start with an empty one)
        table = self.load_table()


        # Step 2: score each specialist and add it as a column
        for name in self.names:
            if name not in table.columns:
                if self.verbose:
                    print(f"  scoring '{name}' on {len(self.reference_queries)} reference queries...")
                description = self.services[name]
                table[name] = self.scorer(self.model_name, self.reference_queries, description)
            elif self.verbose:
                print(f"  '{name}' already in the saved table, skipped")

        # Step 3: keep only the current specialists (in order) and save the table
        self.table = table[self.names]

        # Step 4: create another column called: one lead per reference query (one row of scores -> one number)
        self.leads = self.table.apply(self.shortlist_lead, axis=1)

        saved = self.table.copy()
        saved["lead"] = self.leads
        saved.to_csv(self.cache_file)
        
        # Step 5: the cutoff. percentile=10 -> only the top 10% of off-topic leads pass
        self.cutoff = float(self.leads.quantile(1 - self.percentile / 100))

        if self.verbose:
            print(f"K={len(self.names)}  cutoff={self.cutoff:.3f}")


    def decide(self, scores_dict, fallback="general"):
        """Route ONE query: its shortlist lead must beat self.cutoff, otherwise send it to the fallback."""
        top_name = max(scores_dict, key=scores_dict.get)      # the specialist with the highest score
        lead = self.shortlist_lead(list(scores_dict.values()))
        passed = lead > self.cutoff

        return {
            "label": top_name if passed else fallback,        # what the router should answer
            "top_candidate": top_name,                        # the model's own pick, before the gate
            "lead": float(lead),
            "cutoff": self.cutoff,
            "passed": bool(passed),
        }