# Plotting Spectral Trajectories

A spectral trajectory is the trajectory of a pixel's spectral reflectance or index value over time. When an area has experienced a disturbance (e.g wildfire, clear cut), spectral trajectories can reflect this event due to the change in values typically seen as a sharp drop (the disturbance) followed by a rise (the recovery).

TODO: image with timeseries to traj

A spectral trajectory can be delineated into disturbance and restoration windows to incorporate knowledge of disturbances and subsequent Ecosystem Restoration activities acting on the given pixel. A disturbance window describes the set of years which comprise a disturbance event while a restoration window describes the set of 
years that restoration activities/recovey is taking place.

TODO: timeseries delineated

To support site-specific workflows, `spectral-recovery` provides a method for plotting the spectral trajectory of an entire _restoration site_ instead of just a single pixel. These plots show the mean and median spectral reflectance/index values inside a restoration site over time,   Disturbance and restoration windows, and optionally the recovery target and reference windows, are then displayed with the spectral trajectory. 

TODO: code snippet, output, and diagram

