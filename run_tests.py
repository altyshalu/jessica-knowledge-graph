import os
import sys
import time
import json
import logging
import tempfile
import traceback

# Setup basic logging to suppress noisy external libs
logging.getLogger("urllib3").setLevel(logging.WARNING)

from jkg.memory import HybridMemory

def run_all_tests():
    report = []
    report.append("# JKG 7.1 Memory System Evaluation Report")
    report.append("\n**Tester:** Teston Profile (Automated Evaluation)")
    report.append("**Target:** `jessica-knowledge-graph` (Unified 5-layer memory, SQLite + vectors)")
    report.append("**Embedding Engine:** configured provider or deterministic local fallback\n")
    report.append("This report outlines the execution of 8 edge-case and functionality tests on the memory system.")
    report.append("\n---\n")

    temp_dir = tempfile.TemporaryDirectory()
    test_db_path = os.environ.get(
        "JKG_TEST_DB_PATH",
        os.path.join(temp_dir.name, "jkg-evaluation.sqlite3"),
    )
    hm = HybridMemory(db_path=test_db_path)
    
    # helper for appending
    def add_result(title, status, expected, actual, duration):
        symbol = "✅ PASS" if status else "❌ FAIL"
        report.append(f"### {title}")
        report.append(f"- **Status:** {symbol}")
        report.append(f"- **Expected:** {expected}")
        report.append(f"- **Actual:** {actual}")
        report.append(f"- **Duration:** {duration:.2f}s\n")

    # 1. Multi-Layer Fusion
    start = time.time()
    try:
        hm.remember_profile("User is a software engineer")
        hm.remember("The software engineer built a new compiler")
        hm.remember_session("sess-001", "We discussed compiler design and optimization")
        hm.index_skill("compiler-debug", "Debugs compiler optimization issues", ["optimization", "compiler"])
        
        q_res = hm.query("Tell me about the software engineer and compilers")
        layers = [r.get('layer') for r in q_res.get('results', [])]
        unique_layers = set(filter(None, layers))
        
        if len(unique_layers) > 1:
            add_result("Test 1: Multi-Layer Fusion", True, "Retrieve from multiple layers", f"Retrieved from {unique_layers}", time.time() - start)
        else:
            add_result("Test 1: Multi-Layer Fusion", False, "Retrieve from multiple layers", f"Only retrieved from {unique_layers}", time.time() - start)
    except Exception as e:
        add_result("Test 1: Multi-Layer Fusion", False, "Retrieve from multiple layers", f"Crashed: {str(e)}", time.time() - start)

    # 2. Bi-temporal Validity
    start = time.time()
    try:
        hm.remember("Nikita lived in Bishkek in 2023")
        hm.remember("Nikita moved to San Francisco in 2026")
        res1 = hm.query("Where did Nikita live in 2024?")
        res2 = hm.query("Where does Nikita live in 2026?")
        val1 = " ".join([r.get('value', '') for r in res1['results'][:3]]).lower()
        val2 = " ".join([r.get('value', '') for r in res2['results'][:3]]).lower()
        
        if "bishkek" in val1 and "san francisco" in val2:
            add_result("Test 2: Bi-temporal Validity", True, "Resolve correct locations based on timeline", "Correctly resolved temporal facts", time.time() - start)
        else:
            add_result("Test 2: Bi-temporal Validity", False, "Resolve correct locations based on timeline", f"Res1: {val1[:50]}, Res2: {val2[:50]}", time.time() - start)
    except Exception as e:
        add_result("Test 2: Bi-temporal Validity", False, "Resolve correct locations based on timeline", f"Crashed: {str(e)}", time.time() - start)

    # 3. CLARK Retrieval (Confidence Update)
    start = time.time()
    try:
        hm.remember("Teston's favorite color is Quantum Blue")
        hm.remember("Teston's least favorite color is Rust Red")
        
        # Query color multiple times to boost confidence
        for _ in range(5):
            hm.retrieve_clark("What is Teston's favorite color?")
            
        q_res = hm.retrieve_clark("What is Teston's favorite color?")
        top_fact = q_res[0] if q_res else {}
        
        if "quantum blue" in str(top_fact).lower():
            add_result("Test 3: CLARK Retrieval Confidence", True, "Quantum Blue should rank higher due to retrieval boosting", f"Top fact matched expected. Fact: {top_fact.get('value')}", time.time() - start)
        else:
            add_result("Test 3: CLARK Retrieval Confidence", False, "Quantum Blue should rank higher", f"Top fact was: {top_fact.get('value')}", time.time() - start)
    except Exception as e:
        add_result("Test 3: CLARK Retrieval Confidence", False, "Quantum Blue should rank higher", f"Crashed: {str(e)}", time.time() - start)

    # 4. Conflict Resolution
    start = time.time()
    try:
        hm.remember("Teston hates JavaScript")
        hm.remember("Teston absolutely loves JavaScript")
        
        res = hm.query("How does Teston feel about JavaScript?")
        val = " ".join([r.get('value', '') for r in res['results']]).lower()
        
        if "love" in val and "hate" not in val[:30]: # crude check if newer preference took over
            add_result("Test 4: Conflict Resolution", True, "System should overwrite or prioritize newer conflicting state", "Newer state prioritized", time.time() - start)
        else:
            add_result("Test 4: Conflict Resolution", False, "System should prioritize newer conflicting state", f"Results mixed or failed: {val[:60]}", time.time() - start)
    except Exception as e:
        add_result("Test 4: Conflict Resolution", False, "System should prioritize newer conflicting state", f"Crashed: {str(e)}", time.time() - start)

    # 5. Intentional Forgetting & Utility
    start = time.time()
    try:
        for i in range(10):
            hm.remember(f"I drank coffee cup number {i}")
            
        hm.update_utility_scores()
        prune_res = hm.prune_memory(0.2, dry_run=False) # Prune bottom 20%
        
        add_result("Test 5: Intentional Forgetting", True, "System prunes low-utility nodes without crashing", f"Pruned {prune_res.get('deleted_facts', 0)} facts successfully", time.time() - start)
    except Exception as e:
        add_result("Test 5: Intentional Forgetting", False, "System prunes low-utility nodes without crashing", f"Crashed: {str(e)}", time.time() - start)

    # 6. Self-Evolving Schema
    start = time.time()
    try:
        evolve_res = hm.evolve_schema()
        add_result("Test 6: Self-Evolving Schema", True, "Proposes new schema patterns from existing data", f"Returned {len(evolve_res)} new proposed relations", time.time() - start)
    except Exception as e:
        add_result("Test 6: Self-Evolving Schema", False, "Proposes new schema patterns", f"Crashed: {str(e)}", time.time() - start)

    # 7. GDPR Cascading Delete
    start = time.time()
    try:
        hm.remember("Alice is a secret agent")
        hm.remember("Alice works with Bob")
        hm.gdpr_delete("alice", verified=True)
        
        # Check if Alice exists
        alice_res = hm.query("Who is Alice?")
        if len(alice_res['results']) == 0 or "secret agent" not in str(alice_res).lower():
            add_result("Test 7: GDPR Cascading Delete", True, "All references to Alice are wiped", "Alice wiped successfully", time.time() - start)
        else:
            add_result("Test 7: GDPR Cascading Delete", False, "All references to Alice are wiped", "Traces of Alice remain", time.time() - start)
    except Exception as e:
        add_result("Test 7: GDPR Cascading Delete", False, "All references to Alice are wiped", f"Crashed: {str(e)}", time.time() - start)

    # 8. Semantic Noise
    start = time.time()
    try:
        hm.remember("Apple is a sweet fruit used in pies")
        hm.remember("Apple is a technology company founded by Steve Jobs")
        hm.remember("Apple Records is a record label founded by the Beatles")
        
        res = hm.query("Tell me the recipe for an apple pie")
        val = " ".join([r.get('value', '') for r in res['results'][:3]]).lower()
        entity = " ".join([r.get('entity', '') for r in res['results'][:3]]).lower()
        
        # Food context should dominate: must have fruit/pie terms, must NOT have company/beatles
        food_ok = any(t in val or t in entity for t in ["fruit", "pie", "pies", "sweet"])
        noise_free = "steve jobs" not in val and "beatles" not in val and "technology" not in val and "record label" not in val
        
        if food_ok and noise_free:
            add_result("Test 8: Semantic Noise Injection", True, "Isolates fruit context from tech/music context", "Perfect isolation achieved", time.time() - start)
        else:
            add_result("Test 8: Semantic Noise Injection", False, "Isolates fruit context from tech/music context", f"Mixed contexts retrieved: {val[:60]}", time.time() - start)
    except Exception as e:
        add_result("Test 8: Semantic Noise Injection", False, "Isolates fruit context from tech/music context", f"Crashed: {str(e)}", time.time() - start)

    report_path = os.environ.get("JKG_TEST_REPORT", "jkg_test_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    hm.conn.close()
    temp_dir.cleanup()
    print(f"Tests finished. Report written to {report_path}")

if __name__ == "__main__":
    run_all_tests()
