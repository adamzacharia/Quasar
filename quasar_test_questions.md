# Quasar Test Questions & Follow-ups

This document contains 50 pairs of questions and follow-ups to test Quasar's capabilities, specifically focusing on the new ALminer integration (Search, Visualization, Advanced Queries) and general astronomical data retrieval.

## Basic Target Search

1. **Q:** "Find ALMA observations of Centaurus A."
   * **Follow-up:** "Filter these for Band 6 only."

2. **Q:** "Do you have any data on HL Tau?"
   * **Follow-up:** "What is the total integration time for the longest observation?"

3. **Q:** "Search for observations of the galaxy M87."
   * **Follow-up:** "Are there any high-resolution observations (smaller than 0.1 arcsec)?"

4. **Q:** "Find data for the star Betelgeuse."
   * **Follow-up:** "Show me the project codes associated with these results."

5. **Q:** "Look up observations of Sagittarius A*."
   * **Follow-up:** "Which one has the highest sensitivity?"

6. **Q:** "Find ALMA records for TW Hya."
   * **Follow-up:** "Summarize the frequency bands covered."

7. **Q:** "Search for the protoplanetary disk AS 209."
   * **Follow-up:** "Are any of these public data?"

8. **Q:** "Find observations of NGC 1068."
   * **Follow-up:** "Sort them by observation date, newest first."

9. **Q:** "Look for the quasar 3C 273."
   * **Follow-up:** "What is the bandwidth of the first result?"

10. **Q:** "Search for Eta Carinae."
    * **Follow-up:** "List the Principal Investigators for these proposals."

## Position & Cone Search

11. **Q:** "Find observations within 2 arcmin of RA 10h15m, Dec -30d."
    * **Follow-up:** "Narrow the search radius to 30 arcseconds."

12. **Q:** "Search at RA 12:30:49, Dec +12:23:28 with a 0.05 degree radius."
    * **Follow-up:** "Did this coordinate correspond to a known galaxy?"

13. **Q:** "Are there any ALMA pointings near the Galactic Center?"
    * **Follow-up:** "Plot the sky distribution of these pointings."

14. **Q:** "Perform a cone search around RA 53.1 degrees, Dec -27.8 degrees."
    * **Follow-up:** "Filter for results with integration time > 1000 seconds."

15. **Q:** "Check this location: 19h23m, +14d30m."
    * **Follow-up:** "Are there any Band 3 observations here?"

16. **Q:** "Is there any overlap with the Hubble Ultra Deep Field coordinates?"
    * **Follow-up:** "Show me the most recent observation there."

17. **Q:** "Search near the Orion Nebula Trapezium cluster."
    * **Follow-up:** "Group the results by scientific category."

18. **Q:** "Look for data at RA 200.0, Dec -40.0, radius 0.1 deg."
    * **Follow-up:** "Are there any polarization datasets?"

19. **Q:** "Find sources within 1 arcminute of coordinates 05:35:17 -05:23:28."
    * **Follow-up:** "Create a frequency coverage plot for these."

20. **Q:** "Search the region around coordinates of SN 1987A."
    * **Follow-up:** "Download the metadata for these hits."

## Metadata & Keyword Search (New Features)

21. **Q:** "Find all ALMA proposals by PI 'Smith'."
    * **Follow-up:** "Which of these are from the last 2 years?"

22. **Q:** "Search for project code 2017.1.00001.S."
    * **Follow-up:** "Who is the PI and what is the abstract?"

23. **Q:** "Find observations related to 'protoplanetary disks' in the scientific category."
    * **Follow-up:** "Filter for those in Cycle 7."

24. **Q:** "Show me observations with 'High Mass Star Formation' keyword."
    * **Follow-up:** "Visualize the sky locations of these."

25. **Q:** "Find data collected by 'Remijan' as PI."
    * **Follow-up:** "Download the FITS headers for the first 5 results."

26. **Q:** "Search for proposals mentioning 'magnetic fields' in the abstract."
    * **Follow-up:** "Do any of these have full polarization?"

27. **Q:** "Find all recent public data from Cycle 8."
    * **Follow-up:** "Summarize the main targets observed."

28. **Q:** "Look for observations with resolution better than 0.05 arcsec."
    * **Follow-up:** "Are these mostly Band 7 or Band 6?"

29. **Q:** "Find large project observations (Large Programs)."
    * **Follow-up:** "List the project codes found."

30. **Q:** "Search for data with scan intent 'TARGET' only (exclude calibrators)."
    * **Follow-up:** "How many distinct targets are there?"

## Advanced SQL/TAP Queries

31. **Q:** "Run a query for observations where band=6 and integration_time > 3600."
    * **Follow-up:** "Plot the overview of these results."

32. **Q:** "Find observations where frequency overlaps 230 GHz."
    * **Follow-up:** "Show the frequency coverage plot."

33. **Q:** "Select all public data with sensitivity < 0.5 mJy."
    * **Follow-up:** "Sort by sensitivity."

34. **Q:** "Query for objects within the constellation Orion with Band 9 data."
    * **Follow-up:** "Display the RA/Dec distribution."

35. **Q:** "Find observations with velocity resolution < 1 km/s."
    * **Follow-up:** "Are these suitable for searching for CO lines?"

## Visualization & Analysis

36. **Q:** "Show me a sky map of all ALMA observations of Jupiter."
    * **Follow-up:** "Why are they spread out? (Explain proper motion)."

37. **Q:** "Plot the frequency coverage for 'PDS 70'."
    * **Follow-up:** "Is the CO(3-2) line covered?"

38. **Q:** "Generate an overview plot for the project 2013.1.00099.S."
    * **Follow-up:** "Save this plot."

39. **Q:** "Compare the coverage of Band 3 vs Band 6 for target 'Cyg X-1'."
    * **Follow-up:** "Which band has more integration time?"

40. **Q:** "Visualize the distribution of public ALMA data in the southern sky."
    * **Follow-up:** "Are there any obvious survey fields visible?"

41. **Q:** "Plot the sensitivity vs bandwidth for these results."
    * **Follow-up:** "Interpret the trend in this plot."

42. **Q:** "Create a spectral coverage map for the recent search results."
    * **Follow-up:** "Mark the frequency of the HCN line on the plot."

43. **Q:** "Show the spatial distribution of 'Galactic Centre' pointings."
    * **Follow-up:** "Zoom in on the central parsec if possible."

44. **Q:** "Plot the angular resolution vs frequency."
    * **Follow-up:** "Identify the outliers with highest resolution."

45. **Q:** "Generate a 'footprint' visualization of the mosaics found."
    * **Follow-up:** "How many separate pointings are in this mosaic?"

## Data Access & Downloads

46. **Q:** "Download the FITS files for the observation with UID 'uid://A001/X123/X456'."
    * **Follow-up:** "How large is the downloaded file?"

47. **Q:** "Fetch the first 3 datasets from the search results."
    * **Follow-up:** "Perform a dry run first to check size."

48. **Q:** "Can you get the preview image for this observation?"
    * **Follow-up:** "Is the source resolved in this preview?"

49. **Q:** "Download only the continuum images for 'HL Tau'."
    * **Follow-up:** "Verify the file integrity."

50. **Q:** "Get the QA2 reports for project 2021.1.00001.S."
    * **Follow-up:** "Summarize the QA issues mentioned."

