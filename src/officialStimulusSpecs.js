export const OFFICIAL_PACKAGES = {
  psychophysics: {
    packageName: "@kurokida/jspsych-psychophysics",
    pluginName: "jspsych-psychophysics",
    version: "5.1.0",
    source: "https://github.com/kurokida/jspsych-psychophysics",
  },
  rdk: {
    packageName: "@jspsych-contrib/plugin-rdk",
    pluginName: "rdk",
    version: "2.0.0",
    source: "https://github.com/jspsych/jspsych-contrib/tree/main/packages/plugin-rdk",
  },
};

export function buildMovingGaborSpec(params) {
  return {
    packageName: OFFICIAL_PACKAGES.psychophysics.packageName,
    pluginName: OFFICIAL_PACKAGES.psychophysics.pluginName,
    object: {
      obj_type: "gabor",
      tilt: 90,
      sf: params.spatialFrequency / 960,
      phase: 0,
      width: 960,
      sc: 220,
      contrast: params.contrast,
      drift: params.direction * params.temporalFrequency,
      disableNorm: true,
      disableGauss: false,
    },
  };
}

export function buildLightCircleSpec(params) {
  const bg = Math.round(params.background * 255);
  const spot = Math.round(Math.min(1, params.background + params.increment) * 255);
  return {
    packageName: OFFICIAL_PACKAGES.psychophysics.packageName,
    pluginName: OFFICIAL_PACKAGES.psychophysics.pluginName,
    object: {
      obj_type: "circle",
      radius: Math.round(540 * params.targetSize),
      fill_color: `rgb(${spot}, ${spot}, ${spot})`,
      line_color: `rgb(${spot}, ${spot}, ${spot})`,
      background_color: `rgb(${bg}, ${bg}, ${bg})`,
      show_start_time: params.flashOnset * 1000,
      show_end_time: (params.flashOnset + params.flashDuration) * 1000,
    },
  };
}

export function buildOrientationGaborSpec(params) {
  const leftAngle = params.referenceAngle + (params.differentSide === -1 ? params.deltaAngle : 0);
  const rightAngle = params.referenceAngle + (params.differentSide === 1 ? params.deltaAngle : 0);
  return {
    packageName: OFFICIAL_PACKAGES.psychophysics.packageName,
    pluginName: OFFICIAL_PACKAGES.psychophysics.pluginName,
    objects: [
      {
        obj_type: "gabor",
        startX: 960 * 0.34,
        startY: 540 * 0.52,
        tilt: leftAngle,
        sf: 0.075,
        phase: 0,
        width: 188,
        sc: 49,
        contrast: params.contrast,
        disableNorm: true,
      },
      {
        obj_type: "gabor",
        startX: 960 * 0.66,
        startY: 540 * 0.52,
        tilt: rightAngle,
        sf: 0.075,
        phase: 0,
        width: 188,
        sc: 49,
        contrast: params.contrast,
        disableNorm: true,
      },
    ],
  };
}

export function buildRdkSpec(params) {
  return {
    packageName: OFFICIAL_PACKAGES.rdk.packageName,
    pluginName: OFFICIAL_PACKAGES.rdk.pluginName,
    trial: {
      number_of_dots: params.dotDensity,
      number_of_sets: 1,
      coherent_direction: params.direction,
      coherence: params.coherence,
      opposite_coherence: 0,
      dot_radius: params.dotSize,
      dot_life: -1,
      move_distance: params.moveDistance,
      aperture_width: 454,
      aperture_height: 454,
      aperture_type: 1,
      RDK_type: 3,
      reinsert_type: 2,
      dot_shape: "circle",
      dot_color: "white",
      background_color: "#212733",
      fixation_cross: false,
      border: true,
    },
  };
}
