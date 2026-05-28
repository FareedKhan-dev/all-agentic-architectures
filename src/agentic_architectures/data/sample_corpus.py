"""Sample corpus for RAG-family notebooks (23-27).

A small fictional knowledge base about *Stardust Aerospace*. Made up entirely
so LLM parametric memory can't cheat — any correct answer must come from
retrieval.

Each entry is one document. Mix of short factual statements and slightly
longer narrative paragraphs, like a real corporate knowledge-base scrape.
"""

from __future__ import annotations

STARDUST_CORPUS: list[str] = [
    # Company background
    "Stardust Aerospace was founded in 2019 in Reno, Nevada by Dr. Amara Okonkwo "
    "and Jin-ho Park. The company emerged from a DARPA-funded research program on "
    "low-cost orbital launch systems. As of 2024, Stardust employs 412 people and "
    "operates a manufacturing facility in Sparks, Nevada plus a launch site in the "
    "Black Rock Desert.",
    # Products — Stardust 9 rocket
    "The Stardust 9 is the company's flagship two-stage orbital rocket. It uses "
    "methalox propellant (liquid methane + liquid oxygen) and delivers a maximum "
    "payload of 1,850 kg to low earth orbit (LEO) or 720 kg to geosynchronous "
    "transfer orbit (GTO). First successful orbital flight was in March 2023.",
    # Products — Stardust Lite
    "The Stardust Lite is a smaller single-stage suborbital vehicle introduced "
    "in 2021, primarily used for atmospheric research and short-duration "
    "microgravity experiments. Maximum altitude is 110 km; maximum payload is "
    "240 kg. Stardust Lite has flown 17 missions to date.",
    # Engines
    "Stardust's Phoenix-2 engine powers the first stage of the Stardust 9. It "
    "produces 215 kilonewtons of thrust at sea level and runs on the same "
    "methalox propellant as the rest of the vehicle. The upper-stage Phoenix-1 "
    "engine produces 22 kN of thrust and is vacuum-optimized.",
    # Team / leadership
    "Dr. Amara Okonkwo, Stardust's CEO, holds a PhD in aerospace engineering "
    "from Caltech (2012) and previously worked at Blue Origin on the New Glenn "
    "program. Jin-ho Park, Stardust's CTO and chief engineer, holds an MSc from "
    "KAIST and led propulsion design at the South Korean space agency before "
    "co-founding Stardust.",
    # Launch contracts
    "Stardust Aerospace's primary customers are NOAA, the European Space Agency, "
    "and three commercial smallsat constellation operators. The company has a "
    "$340M multi-launch contract with the National Reconnaissance Office signed "
    "in late 2023, covering up to twelve Stardust 9 missions through 2027.",
    # Safety / mission profile
    "Each Stardust 9 launch follows a strict pre-flight checklist including a "
    "static fire test 72 hours before the planned launch window. The company "
    "operates under FAA Part 450 authorisation, granted in February 2023, with "
    "all flights coordinated through the Reno air route traffic control center.",
    # Cost / pricing
    "Standard list price for a dedicated Stardust 9 launch to LEO is $4.2M, "
    "competitive with Rocket Lab's Electron ($7.5M) on a per-kg basis given "
    "Stardust 9's larger payload. Rideshare missions are priced from $750k for "
    "a 25 kg slot.",
    # Sustainability
    "Stardust's manufacturing facility runs on 100% solar power (10 MW rooftop "
    "array installed in 2022). The company has committed to net-zero ground "
    "operations by 2027 and is researching biofueled launch range vehicles to "
    "further reduce its launch-day footprint.",
    # Recent news
    "In April 2024, Stardust's seventh Stardust 9 launch failed during ascent "
    "due to a Phoenix-2 turbopump anomaly at T+93 seconds. The vehicle was "
    "destroyed; payloads were lost but no injuries occurred. An investigation "
    "led by Jin-ho Park concluded in June 2024 that a single bearing in the "
    "fuel turbopump had been improperly heat-treated by a subcontractor.",
    # Roadmap
    "Stardust's announced 2025-2027 roadmap includes a third vehicle, the "
    "Stardust Heavy, with a target LEO payload of 5,500 kg, scheduled for first "
    "flight no earlier than Q4 2026. The vehicle reuses three Phoenix-2 engines "
    "in the first stage and a new Phoenix-Vacuum upper-stage engine.",
    # HR / culture
    "Stardust's engineering organization is structured into five pods: "
    "propulsion, structures, avionics, ground systems, and mission operations. "
    "The company offers full remote work for software roles but requires "
    "on-site presence at the Sparks facility for hardware engineers. Median "
    "engineering tenure as of mid-2024 is 2.8 years.",
]
