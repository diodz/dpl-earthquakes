Comments to the author (if any):
Reviewer #1: Referee Report
Manuscript Title: Earthquakes and the Wealth of Nations: The cases of Chile and New Zealand Manuscript Number: EMEC-D-25-01824

1. Summary of the Paper
This paper investigates the medium-term regional economic consequences of two major earthquakes: the 2010 Maule event in Chile and the 2010-2011 Canterbury sequence in New Zealand. Using the Synthetic Control Method (SCM), the authors construct counterfactual trajectories for regional GDP per capita to quantify the impact on output.
The study finds a striking divergence in outcomes: in Chile's Maule region, GDP per capita returned to its pre-disaster trend with no significant "overshoot," whereas in New Zealand's Canterbury region, GDP per capita rose approximately 12% above its synthetic counterfactual by 2016. The authors argue that this "Canterbury overshoot" was an institutionally enabled stimulus, facilitated by high insurance penetration (international risk-sharing), decentralized implementation capacity (the CERA authority), and high social capital/community co-production. In contrast, they suggest Chile's centralized response and lower insurance coverage limited the recovery to a mere restoration of the status quo.

2. Major Comments: Story, Narrative, and Contribution

2.1. The "Broken Window Fallacy" and the Wealth vs. Flow Distinction
The paper is titled "Earthquakes and the Wealth of Nations," yet it measures GDP, which is a flow variable.
        Comment: From a conceptual standpoint, replacing destroyed capital stock (houses, roads, businesses) generates a massive spike in GDP (flow) as insurance and government funds are spent. However, this is a replacement of lost wealth, not an addition to it. If a region destroys $30 billion in assets and spends $30 billion to replace them, the region is not "wealthier"; it is likely net-poorer due to the massive opportunity cost of that capital.
        The authors must distinguish between a "replacement boom" and "genuine growth." To justify the "Wealth" narrative, they must prove that Total Factor Productivity (TFP) or productive capacity in non-construction sectors (Manufacturing, Services) increased post-rebuild. Figure 2b shows the gap narrowing toward 2020, suggesting the "gain" was a transitory flow that dissipated once the capital stock was restored.

2.2. Validity of Comparison: Magnitude and Spatial Heterogeneity
The paper treats these events as "comparable," but the physical and geographical nature of the shocks suggests an "apples-to-oranges" problem.
        Comment: The Maule earthquake (8.8  M_w) released orders of magnitude more energy than the Canterbury quakes (M_w  7.1/6.3). More importantly, Maule was a subduction event affecting a vast, linear rural/coastal strip, while Canterbury hit a high-density urban metropolitan hub (Christchurch).
        The Problem: The "null" effect in Chile may simply reflect the logistical difficulty of mobilizing reconstruction across a massive, dispersed geography. Conversely, urban hubs have significantly higher fiscal and consumption multipliers. The authors must control for spatial damage intensity to prove that the difference is institutional rather than purely logistical/geographical.

2.3. Insurance as an Exogenous Capital Inflow
The narrative frames New Zealand's insurance architecture (EQC) as "institutional readiness."
        Comment: From a Macro-perspective, the NZ insurance payout (~$30 billion) was largely funded by international reinsurance. This represents a massive exogenous Balance-of-Payments (BoP) shock, effectively a multi-billion dollar cash gift from global markets to a single city. Chile's recovery was funded by internal domestic reallocation. The "overshoot" is likely a result of international risk-sharing/capital inflows, not just "organized governance." The paper should explicitly discuss this external capital transfer.

3. Major Comments: Data, Method, Results, and Discussion
3.1. The Denominator Effect (Migration and Population Dynamics)
The primary outcome is GDP per capita (Y/L). Large disasters are massive demographic shocks.
        The Bias: If Maule saw a significant out-migration of low-skilled/low-income residents, Y/L would stay constant or rise even if the total economy was in ruins. In Canterbury, if the 12% boost was driven by an influx of thousands of highly-paid itinerant construction workers while original residents fled, the result is a workforce composition shift, not a productivity gain.
The authors must perform separate SCM runs on Total Population and Total Regional GDP. Without proving population stability, the Y/L result is uninterpretable.

3.2. Methodological Rigor: Placebos and Inference
        Systematic In-Time Placebos: The authors report a single falsification test for 2006. They should perform a rolling in-time placebo analysis for every possible year in the pre-treatment period (e.g., 2000-2009). This demonstrates that the 2011 divergence is unique and not a result of pre-existing diverging trends.
        Uniform Confidence Sets: Replace visual "spaghetti plots" with uniform confidence sets (e.g., Firpo & Possebom, 2018). This provides a formal statistical quantification of the uncertainty of the SCM gap over time.

3.3. SCM Estimator Robustness (SDID and Bias-Correction)
        SDID: Given the long post-treatment window (2010-2019), the authors should implement Synthetic Difference-in-Differences (SDID) (Arkhangelsky et al., 2021). SDID is doubly robust and would verify if the "Canterbury overshoot" persists when optimizing both unit and time weights.
        Inexact Matching: Table 2 shows Maule is an agrarian outlier. The authors should apply Bias-Corrected SCM (Abadie & L'Hour, 2021) to mitigate interpolation bias caused by the treated unit being at the edge of the donor pool's convex hull.

3.4. Spatial Spillovers and SUTVA
SCM assumes that the treatment of one unit does not affect the control units.
        The Bias: New Zealand is a small economy (16 regions). A massive rebuild in Canterbury necessarily siphons labor, capital, and machinery from the 15 donor regions. If Auckland's growth slowed because its firms moved to Christchurch, the "Synthetic Counterfactual" is biased downward, artificially inflating the treatment effect. The authors must test for negative spillovers in the donor pool.

3.5. Social Capital and the "Looting" Narrative
The authors argue that "low social trust" and "looting" in Chile hampered recovery.
        Comment: This is speculative and based on national-level surveys. There is no empirical link provided between three days of civil unrest and a decade of GDP stagnation in Maule. Without regional-level trust data, this narrative feels like a stereotypical explanation of an economic residual.

3.6. Circularity and Crowding Out
The paper presents the construction boom as a "mechanism" for stimulus (Figure 5).
        Comment: This is circular reasoning, reconstruction is a construction boom. More importantly, Figure 5b shows a dip in all other sectors. This suggests the construction boom didn't "stimulate" the economy; it cannibalized it. Labor and capital were pulled out of productive sectors to fix broken windows. The authors must address this crowding-out effect.

3.7. The "Narrative Residual" Problem
The authors identify a gap and then "fill" it with institutional storytelling post-hoc.
        Comment: There is no empirical proof that insurance caused the gap. To test the "Insurance Liquidity" channel, the authors should run SCM on Household Consumption or Retail Sales. If the liquidity story is true, we should see a broad-based consumption spike, not just a construction spike.

4. Minor Comments
        Institutional Metrics: Table 4 uses national-level Fraser Institute scores. Applying national averages to regional shocks is weak. Sub-national data on local governance or trust would be more appropriate.
        Circular Reasoning: The sectoral analysis (Figure 5) identifies construction as the mechanism. This is circular, reconstruction is a construction boom (as mentioned earlier). The authors should instead run SCM on Gross Fixed Capital Formation (Investment) or Household Consumption to isolate the liquidity mechanism.
        Data Quality: Corroborating the findings with an objective proxy, such as Nighttime Light (NTL) intensity, would strengthen the paper's credibility by bypassing potential reporting biases in regional GDP accounts during reconstruction (and inclusion of informal economy).




Reviewer #2: This is a well-motivated paper that addresses an important gap in the disaster economics literature. The comparative analysis of two roughly contemporaneous earthquakes in institutionally different contexts is a valuable contribution, and the use of SCM at the regional level is appropriate for this research question. The paper is generally well-written and engages thoughtfully with the relevant literature on natural disasters, institutions, and economic recovery.

The paper contributes to the disaster economics literature by (i) providing the first systematic SCM-based comparison of these two earthquakes, (ii) applying sectoral SCM analysis to examine the composition of economic changes, and (iii) connecting the empirical findings to the broader literature on institutions and disaster recovery.

Major Comments
1.      Identification Concerns and the SUTVA Assumption
The SCM relies on the Stable Unit Treatment Value Assumption (SUTVA), which requires no spillovers between treated and donor units. The authors acknowledge this but address it only partially. For Chile, Region VIII (Biobío) is excluded because it was directly affected, but what about potential economic spillovers to adjacent regions like O'Higgins (Region VI), which receives substantial weight (34.7%) in the synthetic control? If Maule's earthquake displaced economic activity to O'Higgins, this could bias the counterfactual upward, making the null effect appear when there was actually a negative impact. The authors should provide explicit tests or evidence regarding inter-regional trade flows, migration patterns, or input-output linkages. Similarly, for New Zealand, the substantial weight on Auckland (32.7%) raises questions about whether reconstruction-related government spending or labor migration from Auckland to Canterbury could contaminate the counterfactual.
2.      Mechanism Identification and Causal Attribution
The paper's narrative about institutional mechanisms (insurance, governance capacity, social capital) is compelling but remains largely conjectural. The authors correctly acknowledge this limitation in Section 5.2, but this issue is significant enough to warrant more attention. As currently written, the institutional discussion reads as post-hoc rationalization rather than rigorous mechanism analysis. To strengthen this section, the authors should consider: (a) providing quantitative data on insurance claim processing times and amounts in both regions; (b) documenting the timeline of CERA establishment and regulatory changes in New Zealand versus Chile's reconstruction timeline; (c) presenting any available survey evidence on social capital or community participation in the two regions; and (d) acknowledging more explicitly that without micro-data, the mechanism identification remains suggestive rather than definitive.
3.      Comparability of the Two Events
The paper frames the two earthquakes as comparable, but there are important differences that complicate interpretation. The Maule earthquake (Mw 8.8) was vastly larger in physical magnitude than the Canterbury sequence (largest event Mw 7.1), yet the reported damages differ less dramatically ($30B vs. $15B). The Canterbury sequence involved multiple events over 2010-2011, creating sustained uncertainty, whereas Maule was a single shock. These differences may affect the economic response independently of institutions. The authors should: (a) discuss how the multi-shock nature of Canterbury might have created different investment incentives; (b) consider whether the "18% of GDP" vs. "10% of GDP" damage estimates are measured consistently; and (c) address whether the Maule region's more rural character compared to Christchurch's urban economy could explain different recovery dynamics.
4.      Sectoral Analysis Presentation and Interpretation
The sectoral SCM analysis (Section 4.2) is an interesting methodological contribution, but its presentation is incomplete. The Maule sectoral results are described as showing "no deviations" but are not shown "for brevity." Given that this is a comparative study, the Maule sectoral figures should be included—either in the main text or an appendix. Additionally, the claim that Canterbury's construction sector "nearly doubled" its share of GDP should be made more precise. Figure 5a shows construction rising from ~6% to ~10%, which is a 67% increase in share, not a doubling. Finally, statistical inference for the sectoral SCM (placebo tests for construction share) should be provided.
5.      Alternative Explanations and Confounders
The paper attributes the divergent outcomes primarily to institutional factors, but several alternative explanations deserve more attention: (a) Global commodity prices: Chile's Maule region is heavily agricultural (16.5% of GDP). Were there commodity price movements during 2010-2015 that affected agricultural regions differently? (b) Exchange rate movements: Chile and New Zealand experienced different exchange rate trajectories post-2010. How might this have affected regional competitiveness? (c) The Canterbury earthquakes coincided with the Global Financial Crisis recovery period—might the reconstruction have substituted for other stimulus that would have occurred anyway? (d) The 2014 Iquique earthquake affected Region I in Chile. Although the authors mention this, they should verify that excluding this region does not change the Maule results.


4. Minor Comments
1.      Predictor Selection
Table 2 presents predictor means, but the paper does not discuss how predictor weights (vm) were chosen. Did the authors use the data-driven approach of Abadie et al. (2010) or an alternative? The choice can affect results. Additionally, why is "Fishing" included for Chile but not New Zealand? This asymmetry should be explained.
2.      Treatment Timing Clarification
The treatment timing for Canterbury needs clarification. The September 2010 Darfield earthquake preceded the destructive February 2011 event. The authors treat 2011 as the treatment year but should provide sensitivity analysis using 2010 as the treatment year (they mention doing so but do not show results). This matters because the pre-treatment fit could differ.
3.      Welfare Interpretation Caveat
The authors note in passing (Section 4.1) that GDP gains do not imply welfare gains. This important caveat should be moved to a more prominent position, perhaps in the introduction or conclusion. The Canterbury "boom" came at the cost of 185 lives and substantial displacement—readers should not interpret the positive GDP effect as suggesting earthquakes are beneficial.
4.      Table 4 Integration
Table 4 (fiscal and institutional comparison) appears late in the paper (Section 5) and is described as containing "illustrative descriptors rather than inputs in the SCM." If these variables are important for the institutional interpretation, consider whether they could be incorporated into the analysis more formally—for example, by discussing whether differences in government debt capacity might have constrained Chile's reconstruction response.