"""
Comprehensive Python port of the CUMBIA MATLAB scripts.

The module reproduces the rectangular and circular reinforced-concrete member
analysis workflows from the original MATLAB implementation using NumPy,
SciPy-style interpolation, and Matplotlib for visualization. All variables use
lower camelCase naming to match the user request and ease direct comparison
with the MATLAB source.

Example
-------
Run a circular section analysis with the default example parameters::

    from cumbia import runCircularExample
    circularResult = runCircularExample()
    print(f"Ultimate moment = {circularResult['moments'][-1]:.2f} kN-m")

Run a rectangular section analysis with the default example parameters::

    from cumbia import runRectangularExample
    rectResult = runRectangularExample()
    print(f"Ultimate displacement = {rectResult['displacements'][-1]:.4f} m")

Both helpers generate plots and write a text summary that mirrors the original
``.xls`` output files. For custom studies, build a parameter dictionary (see
``defaultCircularConfig`` and ``defaultRectangularConfig``) and pass it to
``analyzeCircularSection`` or ``analyzeRectangularSection``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Material model utilities
# --------------------------------------------------------------------------- #


def manderUnconfined(
    ecModulus: float,
    steelAreaTotal: float,
    hoopDiameter: float,
    coverToLongBars: float,
    transverseSpacing: float,
    concreteStrength: float,
    transverseYield: float,
    eco: float,
    esm: float,
    espAll: float,
    sectionType: str,
    diameter: float,
    depth: float,
    width: float,
    hoopsX: int,
    hoopsY: int,
    wiVector: Iterable[float],
    strainStep: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the unconfined concrete stress-strain curve using the Mander model.

    Parameters are kept close to the MATLAB signature for traceability.

    Returns
    -------
    strains : np.ndarray
        Strain history starting at zero up to ``espAll``.
    stresses : np.ndarray
        Stress history in MPa with the same length as ``strains``.
    """

    strains = np.arange(0, espAll + strainStep, strainStep)
    eseCu = concreteStrength / eco
    ru = ecModulus / (ecModulus - eseCu)
    xu = strains / eco
    stresses: List[float] = []

    for strain in strains:
        if strain < 2 * eco:
            stresses.append(concreteStrength * (strain / eco) * ru / (ru - 1 + (strain / eco) ** ru))
        elif 2 * eco <= strain <= espAll:
            stresses.append(concreteStrength * (2 * ru / (ru - 1 + 2**ru)) * (1 - (strain - 2 * eco) / (espAll - 2 * eco)))
        else:
            stresses.append(0.0)

    return strains, np.asarray(stresses)


def manderConfined(
    ecModulus: float,
    steelAreaTotal: float,
    hoopDiameter: float,
    coverToLongBars: float,
    transverseSpacing: float,
    concreteStrength: float,
    transverseYield: float,
    eco: float,
    esm: float,
    espAll: float,
    sectionType: str,
    diameter: float,
    depth: float,
    width: float,
    hoopsX: int,
    hoopsY: int,
    wiVector: Iterable[float],
    strainStep: float,
    transverseType: str,
    lightweight: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Confined concrete stress-strain curve following the Mander model.

    The ``lightweight`` flag toggles the lightweight variant used in the
    original ``mclw`` MATLAB helper.
    """

    spacingClear = transverseSpacing - hoopDiameter
    hoopArea = 0.25 * math.pi * hoopDiameter**2

    if sectionType.lower() == "rectangular":
        bc = width - 2 * coverToLongBars + hoopDiameter
        dc = depth - 2 * coverToLongBars + hoopDiameter
        asx = hoopsX * hoopArea
        asy = hoopsY * hoopArea
        ac = bc * dc
        rocc = steelAreaTotal / ac
        rox = asx / (transverseSpacing * dc)
        roy = asy / (transverseSpacing * bc)
        ros = rox + roy
        ke = ((1 - np.sum(np.square(wiVector)) / (6 * bc * dc)) * (1 - spacingClear / (2 * bc)) * (1 - spacingClear / (2 * dc))) / (1 - rocc)
        ro = 0.5 * ros
        fpl = ke * ro * transverseYield
    elif sectionType.lower() == "circular":
        ds = diameter - 2 * coverToLongBars + hoopDiameter
        ros = 4 * hoopArea / (ds * transverseSpacing)
        ac = 0.25 * math.pi * ds**2
        rocc = steelAreaTotal / ac
        if transverseType.lower() == "spirals":
            ke = (1 - spacingClear / (2 * ds)) / (1 - rocc)
        elif transverseType.lower() == "hoops":
            ke = ((1 - spacingClear / (2 * ds)) / (1 - rocc)) ** 2
        else:
            raise ValueError("transverseType should be 'spirals' or 'hoops'")
        fpl = 0.5 * ke * ros * transverseYield
    else:
        raise ValueError("sectionType must be 'rectangular' or 'circular'")

    if lightweight:
        fpcc = (1 + fpl / (2 * concreteStrength)) * concreteStrength
    else:
        fpcc = (-1.254 + 2.254 * math.sqrt(1 + 7.94 * fpl / concreteStrength) - 2 * fpl / concreteStrength) * concreteStrength

    ecc = eco * (1 + 5 * (fpcc / concreteStrength - 1))
    eSec = fpcc / ecc
    rFactor = ecModulus / (ecModulus - eSec)
    ecu = 1.5 * (0.004 + 1.4 * (ros * transverseYield * esm) / fpcc)

    strains = np.arange(0, ecu + strainStep, strainStep)
    normalized = strains / ecc
    stresses = fpcc * normalized * rFactor / (rFactor - 1 + np.power(normalized, rFactor))
    return strains, stresses


def raynorSteelModel(
    elasticModulus: float,
    yieldStress: float,
    ultimateStress: float,
    strainHardeningStrain: float,
    ultimateStrain: float,
    strainStep: float,
    c1: float,
    yieldPlateauSlope: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raynor steel model translation.
    """

    strains = np.arange(0, ultimateStrain + strainStep, strainStep)
    yieldStrain = yieldStress / elasticModulus
    fsh = yieldStress + (strainHardeningStrain - yieldStrain) * yieldPlateauSlope
    stresses: List[float] = []
    denominator = max(ultimateStrain - strainHardeningStrain, 1e-12)

    for strain in strains:
        if strain < yieldStrain:
            stresses.append(elasticModulus * strain)
        elif yieldStrain <= strain <= strainHardeningStrain:
            stresses.append(yieldStress + (strain - yieldStrain) * yieldPlateauSlope)
        else:
            ratio = (ultimateStrain - strain) / denominator
            ratio = max(0.0, min(1.0, ratio))
            stresses.append(ultimateStress - (ultimateStress - fsh) * (ratio**c1))

    return strains, np.asarray(stresses)


def steelKingModel(
    elasticModulus: float,
    yieldStress: float,
    ultimateStress: float,
    strainHardeningStrain: float,
    ultimateStrain: float,
    strainStep: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Steel King model translation.
    """

    rValue = ultimateStrain - strainHardeningStrain
    mValue = ((ultimateStress / yieldStress) * ((30 * rValue + 1) ** 2) - 60 * rValue - 1) / (15 * (rValue**2))
    strains = np.arange(0, ultimateStrain + strainStep, strainStep)
    yieldStrain = yieldStress / elasticModulus
    stresses: List[float] = []

    for strain in strains:
        if strain < yieldStrain:
            stresses.append(elasticModulus * strain)
        elif yieldStrain <= strain <= strainHardeningStrain:
            stresses.append(yieldStress)
        else:
            stresses.append(((mValue * (strain - strainHardeningStrain) + 2) / (60 * (strain - strainHardeningStrain) + 2) + (strain - strainHardeningStrain) * (60 - mValue) / (2 * ((30 * rValue + 1) ** 2))) * yieldStress)

    return strains, np.asarray(stresses)


def mirrorSteelCurve(strains: np.ndarray, stresses: np.ndarray, strainStep: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror a tension-only steel curve to compression to mimic MATLAB concatenation.
    """

    maxStrain = strains[-1]
    extendedStrains = np.concatenate([strains, np.array([maxStrain + strainStep, 1e10])])
    extendedStresses = np.concatenate([stresses, np.array([0.0, 0.0])])
    mirroredStrains = -np.flip(extendedStrains)
    mirroredStresses = -np.flip(extendedStresses)
    fullStrains = np.concatenate([mirroredStrains[:-1], extendedStrains])
    fullStresses = np.concatenate([mirroredStresses[:-1], extendedStresses])
    return fullStrains, fullStresses


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def circularLayerAreas(
    diameter: float,
    coreDiameter: float,
    coverToLongBars: float,
    hoopDiameter: float,
    layerCount: int,
    rebarAreas: np.ndarray,
    rebarDistances: np.ndarray,
) -> np.ndarray:
    """
    Build the concrete layer matrix for circular sections.
    """

    layerThickness = diameter / layerCount
    layerBorders = layerThickness * np.arange(1, layerCount + 1)
    layerBorders = np.sort(np.concatenate([layerBorders, np.array([coverToLongBars - hoopDiameter * 0.5, diameter - (coverToLongBars - hoopDiameter * 0.5)])]))
    condensed: List[float] = []
    for idx in range(len(layerBorders) - 1):
        if layerBorders[idx] != layerBorders[idx + 1]:
            condensed.append(layerBorders[idx])
    layerBorders = np.array(condensed + [layerBorders[-1]])
    confinedBorders = layerBorders - (coverToLongBars - hoopDiameter * 0.5)
    confinedBorders = np.concatenate([confinedBorders[(confinedBorders > 0) & (confinedBorders < coreDiameter)], np.array([coreDiameter])])

    areaAux = (0.25 * math.pi * diameter**2) * np.arccos(1 - 2 * layerBorders / diameter) - (diameter / 2 - layerBorders) * np.sqrt(np.maximum(diameter * layerBorders - layerBorders**2, 0))
    totalAreas = areaAux - np.concatenate([np.array([0.0]), areaAux[:-1]])

    coreAreaAux = (0.25 * math.pi * coreDiameter**2) * np.arccos(1 - 2 * confinedBorders / coreDiameter) - (coreDiameter / 2 - confinedBorders) * np.sqrt(np.maximum(coreDiameter * confinedBorders - confinedBorders**2, 0))
    coreAreas = coreAreaAux - np.concatenate([np.array([0.0]), coreAreaAux[:-1]])

    layerData: List[Tuple[float, float, float]] = []
    coreIndex = 0
    for border in layerBorders:
        if border <= coverToLongBars - hoopDiameter * 0.5 or border > diameter - (coverToLongBars - hoopDiameter * 0.5):
            layerData.append((border, totalAreas[len(layerData)], 0.0))
        else:
            layerData.append((border, totalAreas[len(layerData)] - coreAreas[coreIndex], coreAreas[coreIndex]))
            coreIndex += 1

    layerMatrix = np.array(layerData)
    centers = np.concatenate([[layerBorders[0] / 2], 0.5 * (layerBorders[:-1] + layerBorders[1:])])
    layerMatrix = np.column_stack([centers, layerMatrix[:, 1:], layerBorders])
    layerMatrix = np.column_stack([layerMatrix, np.zeros(layerMatrix.shape[0])])

    for idx in range(1, layerMatrix.shape[0] - 1):
        mask = (rebarDistances <= layerMatrix[idx, 3]) & (rebarDistances > layerMatrix[idx - 1, 3])
        layerMatrix[idx, 4] = np.sum(rebarAreas[mask])
        if layerMatrix[idx, 2] == 0:
            layerMatrix[idx, 1] = max(0.0, layerMatrix[idx, 1] - layerMatrix[idx, 4])
        else:
            layerMatrix[idx, 2] = max(0.0, layerMatrix[idx, 2] - layerMatrix[idx, 4])

    return layerMatrix


def rectangularLayerAreas(
    height: float,
    width: float,
    coreHeight: float,
    coreWidth: float,
    coverToLongBars: float,
    hoopDiameter: float,
    layerCount: int,
    rebarAreas: np.ndarray,
    rebarDistances: np.ndarray,
) -> np.ndarray:
    """
    Build the concrete layer matrix for rectangular sections.
    """

    layerThickness = height / layerCount
    layerBorders = layerThickness * np.arange(1, layerCount + 1)
    layerBorders = np.sort(np.concatenate([layerBorders, np.array([coverToLongBars - hoopDiameter * 0.5, height - (coverToLongBars - hoopDiameter * 0.5)])]))
    condensed: List[float] = []
    for idx in range(len(layerBorders) - 1):
        if layerBorders[idx] != layerBorders[idx + 1]:
            condensed.append(layerBorders[idx])
    layerBorders = np.array(condensed + [layerBorders[-1]])
    confinedBorders = layerBorders - (coverToLongBars - hoopDiameter * 0.5)
    confinedBorders = np.concatenate([confinedBorders[(confinedBorders > 0) & (confinedBorders < coreHeight)], np.array([coreHeight])])

    totalAreas = np.concatenate([[layerBorders[0]], np.diff(layerBorders)]) * width
    coreAreas = np.concatenate([[confinedBorders[0]], np.diff(confinedBorders)]) * coreWidth

    layerData: List[Tuple[float, float, float]] = []
    coreIndex = 0
    for border in layerBorders:
        if border <= coverToLongBars - hoopDiameter * 0.5 or border > height - (coverToLongBars - hoopDiameter * 0.5):
            layerData.append((border, totalAreas[len(layerData)], 0.0))
        else:
            layerData.append((border, totalAreas[len(layerData)] - coreAreas[coreIndex], coreAreas[coreIndex]))
            coreIndex += 1

    layerMatrix = np.array(layerData)
    centers = np.concatenate([[layerBorders[0] / 2], 0.5 * (layerBorders[:-1] + layerBorders[1:])])
    layerMatrix = np.column_stack([centers, layerMatrix[:, 1:], layerBorders])
    layerMatrix = np.column_stack([layerMatrix, np.zeros(layerMatrix.shape[0])])

    for idx in range(1, layerMatrix.shape[0] - 1):
        mask = (rebarDistances <= layerMatrix[idx, 3]) & (rebarDistances > layerMatrix[idx - 1, 3])
        layerMatrix[idx, 4] = np.sum(rebarAreas[mask])
        if layerMatrix[idx, 2] == 0:
            layerMatrix[idx, 1] = max(0.0, layerMatrix[idx, 1] - layerMatrix[idx, 4])
        else:
            layerMatrix[idx, 2] = max(0.0, layerMatrix[idx, 2] - layerMatrix[idx, 4])

    return layerMatrix


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #


def buildSteelModel(
    modelName: str,
    elasticModulus: float,
    yieldStress: float,
    ultimateStress: float,
    strainHardeningStrain: float,
    ultimateStrain: float,
    strainStep: float,
    c1: float,
    yieldPlateauSlope: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the requested steel model or load a custom curve from disk.
    """

    lowerName = modelName.lower()
    if lowerName == "ks":
        strains, stresses = steelKingModel(elasticModulus, yieldStress, ultimateStress, strainHardeningStrain, ultimateStrain, strainStep)
    elif lowerName == "ra":
        strains, stresses = raynorSteelModel(elasticModulus, yieldStress, ultimateStress, strainHardeningStrain, ultimateStrain, strainStep, c1, yieldPlateauSlope)
    else:
        data = np.loadtxt(Path("models") / f"{modelName}.txt")
        strains = data[:, 0]
        stresses = data[:, 1]
    return mirrorSteelCurve(strains, stresses, strainStep)


def buildConcreteModel(
    modelName: str,
    ecModulus: float,
    steelAreaTotal: float,
    hoopDiameter: float,
    coverToLongBars: float,
    transverseSpacing: float,
    concreteStrength: float,
    transverseYield: float,
    eco: float,
    esm: float,
    espAll: float,
    sectionType: str,
    diameter: float,
    depth: float,
    width: float,
    hoopsX: int,
    hoopsY: int,
    wiVector: Iterable[float],
    strainStep: float,
    transverseType: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the requested concrete model or load a custom curve from disk.
    """

    lowerName = modelName.lower()
    if lowerName == "mu":
        strains, stresses = manderUnconfined(ecModulus, steelAreaTotal, hoopDiameter, coverToLongBars, transverseSpacing, concreteStrength, transverseYield, eco, esm, espAll, sectionType, diameter, depth, width, hoopsX, hoopsY, wiVector, strainStep)
    elif lowerName == "mc":
        strains, stresses = manderConfined(ecModulus, steelAreaTotal, hoopDiameter, coverToLongBars, transverseSpacing, concreteStrength, transverseYield, eco, esm, espAll, sectionType, diameter, depth, width, hoopsX, hoopsY, wiVector, strainStep, transverseType, lightweight=False)
    elif lowerName == "mclw":
        strains, stresses = manderConfined(ecModulus, steelAreaTotal, hoopDiameter, coverToLongBars, transverseSpacing, concreteStrength, transverseYield, eco, esm, espAll, sectionType, diameter, depth, width, hoopsX, hoopsY, wiVector, strainStep, transverseType, lightweight=True)
    else:
        data = np.loadtxt(Path("models") / f"{modelName}.txt")
        strains, stresses = data[:, 0], data[:, 1]
    extendedStrains = np.concatenate([[-1e10], strains, [strains[-1] + strainStep, 1e10]])
    extendedStresses = np.concatenate([[0.0], stresses, [0.0, 0.0]])
    return extendedStrains, extendedStresses


def extendSteelForCompression(strains: np.ndarray, stresses: np.ndarray, strainStep: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrically extend the steel model for compression.
    """

    mirrored = np.flip(strains)
    mirroredStress = np.flip(stresses)
    mirrored = -mirrored
    mirroredStress = -mirroredStress
    combinedStrains = np.concatenate([mirrored[:-1], strains[1:]])
    combinedStresses = np.concatenate([mirroredStress[:-1], stresses[1:]])
    return combinedStrains, combinedStresses


def interpolation(values: np.ndarray, xs: np.ndarray, xPoints: np.ndarray) -> np.ndarray:
    """
    Wrapper around NumPy interpolation to mimic MATLAB's interp1 with linear fill.
    """

    return np.interp(xPoints, xs, values)


def buildDeflectionVector(maxStrain: float, strainStep: float) -> np.ndarray:
    """
    Recreate the piecewise strain vector used in MATLAB to drive the analysis.
    """

    if maxStrain <= 0.0018:
        return np.arange(0.0001, 20 * maxStrain + 0.0001, 0.0001)
    if 0.0018 < maxStrain <= 0.0025:
        return np.concatenate([np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 20 * maxStrain + strainStep, 0.0002)])
    if 0.0025 < maxStrain <= 0.006:
        return np.concatenate([np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0021, 0.0002), np.arange(0.0025, 20 * maxStrain + strainStep, 0.0005)])
    if 0.006 < maxStrain <= 0.012:
        return np.concatenate(
            [
                np.arange(0.0001, 0.0017, 0.0001),
                np.arange(0.0018, 0.0021, 0.0002),
                np.arange(0.0025, 0.0051, 0.0005),
                np.arange(0.006, 20 * maxStrain + strainStep, 0.001),
            ]
        )
    return np.concatenate(
        [
            np.arange(0.0001, 0.0017, 0.0001),
            np.arange(0.0018, 0.0021, 0.0002),
            np.arange(0.0025, 0.0051, 0.0005),
            np.arange(0.006, 0.0101, 0.001),
            np.arange(0.012, 20 * maxStrain + strainStep, 0.002),
        ]
    )


# --------------------------------------------------------------------------- #
# Core analysis engine shared by shapes
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisResults:
    moments: np.ndarray
    curvatures: np.ndarray
    neutralAxes: np.ndarray
    coverStrains: np.ndarray
    coreStrains: np.ndarray
    steelStrains: np.ndarray
    forces: np.ndarray
    shearDisplacements: np.ndarray
    flexuralDisplacements: np.ndarray
    displacements: np.ndarray
    shearCapacityAssess: np.ndarray
    shearCapacityDesign: np.ndarray
    bilinearCurvatures: np.ndarray
    bilinearMoments: np.ndarray
    bilinearDisplacements: np.ndarray
    bilinearForces: np.ndarray
    additional: Dict[str, float] = field(default_factory=dict)


def computeMomentCurvature(
    sectionLabel: str,
    defVector: np.ndarray,
    tolerance: float,
    iterMax: int,
    axialLoad: float,
    conclay: np.ndarray,
    rebarDistances: np.ndarray,
    rebarArea: np.ndarray,
    concreteUnconfined: Tuple[np.ndarray, np.ndarray],
    concreteConfined: Tuple[np.ndarray, np.ndarray],
    steelModel: Tuple[np.ndarray, np.ndarray],
    maxSteelStrain: float,
    maxConcreteStrain: float,
    depthReference: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Iterate over top-fiber strains to build moment-curvature response.
    """

    ecUn, fcUn = concreteUnconfined
    ecCo, fcCo = concreteConfined
    esVals, fsVals = steelModel

    curvatures = [0.0]
    moments = [0.0]
    neutralAxes = [0.0]
    dfHistory = [0.0]
    iterations = [0]
    coverStrains = [0.0]
    coreStrains = [0.0]
    steelStrains = [0.0]
    neutralAxis = depthReference / 2
    messageFlag = 0

    for idx, topStrain in enumerate(defVector, start=1):
        if messageFlag:
            break
        lostControl = max(moments)
        if moments[-1] < 0.8 * lostControl:
            messageFlag = 4
            break
        forceEquilibrium = 10 * tolerance
        iteration = 0
        while abs(forceEquilibrium) > tolerance:
            iteration += 1
            concreteStrains = (topStrain / neutralAxis) * (conclay[:, 0] - (depthReference - neutralAxis))
            steelStrainsCurrent = (topStrain / neutralAxis) * (rebarDistances - (depthReference - neutralAxis))
            fcUnconfined = interpolation(fcUn, ecUn, concreteStrains)
            fcConfined = interpolation(fcCo, ecCo, concreteStrains)
            steelStresses = interpolation(fsVals, esVals, steelStrainsCurrent)
            funcon = fcUnconfined * conclay[:, 1]
            fconf = fcConfined * conclay[:, 2]
            fsteel = rebarArea * steelStresses
            forceEquilibrium = np.sum(funcon) + np.sum(fconf) + np.sum(fsteel) - axialLoad
            if forceEquilibrium > 0:
                neutralAxis -= 0.05 * neutralAxis
            elif forceEquilibrium < 0:
                neutralAxis += 0.05 * neutralAxis
            if iteration > iterMax:
                messageFlag = 3
                break
        coreStrain = (topStrain / neutralAxis) * abs(neutralAxis - (conclay[0, 3] - conclay[0, 0]))
        if coreStrain >= maxConcreteStrain:
            messageFlag = 1
        if abs(steelStrainsCurrent[0]) > maxSteelStrain:
            messageFlag = 2

        neutralAxes.append(neutralAxis)
        dfHistory.append(forceEquilibrium)
        iterations.append(iteration)
        currentMoment = (np.sum(funcon * conclay[:, 0]) + np.sum(fconf * conclay[:, 0]) + np.sum(fsteel * rebarDistances) - axialLoad * (depthReference / 2)) / 1e6
        if currentMoment < 0:
            currentMoment = -0.01 * currentMoment
        moments.append(currentMoment)
        curvature = 1000 * topStrain / neutralAxis
        curvatures.append(curvature)
        coverStrains.append(topStrain)
        coreStrains.append(coreStrain)
        steelStrains.append(steelStrainsCurrent[0])
        if messageFlag:
            break
    return (
        np.asarray(moments),
        np.asarray(curvatures),
        np.asarray(neutralAxes),
        np.asarray(coverStrains),
        np.asarray(coreStrains),
        np.asarray(steelStrains),
        messageFlag,
    )


# --------------------------------------------------------------------------- #
# Rectangular analysis
# --------------------------------------------------------------------------- #


def analyzeRectangularSection(config: Dict) -> AnalysisResults:
    """
    Execute the rectangular CUMBIA workflow.
    """

    name = config["name"]
    interaction = config["interaction"]
    height = config["height"]
    width = config["width"]
    hoopsX = config["hoopsX"]
    hoopsY = config["hoopsY"]
    coverToLongBars = config["coverToLongBars"]
    memberLength = config["memberLength"]
    bending = config["bending"]
    ductilityMode = config["ductilityMode"]
    longitudinalRows = np.array(config["longitudinalRows"], dtype=float)
    hoopDiameter = config["hoopDiameter"]
    transverseSpacing = config["transverseSpacing"]
    axialLoad = config["axialLoad"] * 1000
    confinedModelName = config["confinedModel"]
    unconfinedModelName = config["unconfinedModel"]
    steelModelName = config["steelModel"]
    wiVector = np.array(config["wiVector"], dtype=float)
    fpc = config["fpc"]
    ecModulus = 5000 * math.sqrt(fpc) if config["ecModulus"] == 0 else config["ecModulus"]
    eco = config["eco"]
    esm = config["esm"]
    espAll = config["espAll"]
    fy = config["fy"]
    fyh = config["fyh"]
    elasticModulusSteel = config["elasticModulusSteel"]
    fsu = config["fsu"]
    esh = config["esh"]
    esu = config["esu"]
    eySlope = config["eySlope"]
    c1Value = config["c1Value"]
    csid = config["csid"]
    ssid = config["ssid"]
    ecser = config["ecser"]
    esser = -config["esser"]
    ecdam = config["ecdam"]
    esdam = -config["esdam"]
    temp = config["temperature"]
    kLsp = config["kLsp"]
    iterMax = config["iterMax"]
    ncl = config["ncl"]
    tolerance = config["tolerance"]
    strainStep = config["strainStep"]

    if wiVector.sum() == 0 and longitudinalRows.shape[0] > 1:
        pBars = longitudinalRows[0, 1] + longitudinalRows[-1, 1] + 2 * (longitudinalRows.shape[0] - 2)
        wiVector = np.concatenate(
            [
                np.full(int(longitudinalRows[0, 1] - 1), (width - 2 * coverToLongBars - longitudinalRows[0, 1] * longitudinalRows[0, 2]) / (longitudinalRows[0, 1] - 1)),
                np.full(int(longitudinalRows[-1, 1] - 1), (width - 2 * coverToLongBars - longitudinalRows[-1, 1] * longitudinalRows[-1, 2]) / (longitudinalRows[-1, 1] - 1)),
                longitudinalRows[1:, 0] - longitudinalRows[:-1, 0] - np.mean(longitudinalRows[:, 2]),
                longitudinalRows[1:, 0] - longitudinalRows[:-1, 0] - np.mean(longitudinalRows[:, 2]),
            ]
        )
    if wiVector.sum() == 0 and longitudinalRows.shape[0] == 1:
        wiVector = np.array([(height - 2 * coverToLongBars - 2 * longitudinalRows[0, 2]), (height - 2 * coverToLongBars - 2 * longitudinalRows[0, 2]), (width - 2 * coverToLongBars - 2 * longitudinalRows[0, 2]), (width - 2 * coverToLongBars - 2 * longitudinalRows[0, 2])])

    ast = np.sum(longitudinalRows[:, 1] * 0.25 * math.pi * longitudinalRows[:, 2] ** 2)
    heightCore = height - 2 * coverToLongBars + hoopDiameter
    widthCore = width - 2 * coverToLongBars + hoopDiameter
    dcore = coverToLongBars - hoopDiameter * 0.5

    concreteUnconfined = buildConcreteModel(unconfinedModelName, ecModulus, ast, hoopDiameter, coverToLongBars, transverseSpacing, fpc, fyh, eco, esm, espAll, "rectangular", 0, height, width, hoopsX, hoopsY, wiVector, strainStep, "hoops")
    concreteConfined = buildConcreteModel(confinedModelName, ecModulus, ast, hoopDiameter, coverToLongBars, transverseSpacing, fpc, fyh, eco, esm, espAll, "rectangular", 0, height, width, hoopsX, hoopsY, wiVector, strainStep, "hoops")
    steelModel = buildSteelModel(steelModelName, elasticModulusSteel, fy, fsu, esh, esu, strainStep, c1Value, eySlope)
    esuLimit = steelModel[0][-2]
    if isinstance(ecdam, str) and ecdam.lower() == "twth":
        ecdam = (concreteConfined[0][-2]) / 1.5

    rebarDistances, rebarAreas, barDiameters = [], [], []
    for row in longitudinalRows:
        distances = np.full(int(row[1]), row[0])
        rebarDistances.extend(distances)
        rebarAreas.extend(np.full(int(row[1]), 0.25 * math.pi * row[2] ** 2))
        barDiameters.extend(np.full(int(row[1]), row[2]))
    rebarDistances = np.array(rebarDistances)
    rebarAreas = np.array(rebarAreas)
    barDiameters = np.array(barDiameters)
    sortedOrder = np.argsort(rebarDistances)
    rebarDistances = rebarDistances[sortedOrder]
    rebarAreas = rebarAreas[sortedOrder]
    barDiameters = barDiameters[sortedOrder]

    conclay = rectangularLayerAreas(height, width, heightCore, widthCore, coverToLongBars, hoopDiameter, ncl, rebarAreas, rebarDistances)
    defVector = buildDeflectionVector(concreteConfined[0][-2], strainStep)

    if axialLoad > 0:
        for idx in range(len(defVector)):
            compCheck = np.sum(interpolation(concreteUnconfined[1], concreteUnconfined[0], defVector[0] * np.ones(len(conclay))) * conclay[:, 1]) + np.sum(interpolation(concreteConfined[1], concreteConfined[0], defVector[0] * np.ones(len(conclay))) * conclay[:, 2]) + np.sum(ast * interpolation(steelModel[1], steelModel[0], defVector[0]))
            if compCheck < axialLoad:
                defVector = defVector[1:]
            else:
                break

    moments, curvatures, neutralAxes, coverStrains, coreStrains, steelStrains, messageFlag = computeMomentCurvature(
        "rectangular",
        defVector,
        tolerance * height * width * fpc,
        iterMax,
        axialLoad,
        conclay,
        rebarDistances,
        rebarAreas,
        concreteUnconfined,
        concreteConfined,
        steelModel,
        esuLimit,
        concreteConfined[0][-2],
        height,
    )

    mn = np.interp(ecser, coverStrains, moments)
    esaux = np.interp(ecser, coverStrains, steelStrains)
    cr = 0
    if abs(esaux) > abs(esser) or np.isnan(mn):
        mnSteel = np.interp(esser, steelStrains, moments)
        if not math.isnan(mnSteel):
            cr = 1
            mn = mnSteel
        else:
            raise RuntimeError("Problem estimating nominal moment")
    cMn = np.interp(mn, moments, neutralAxes)
    fycurvC = np.interp(1.8 * fpc / ecModulus, coverStrains, curvatures)
    fycurvS = np.interp(-fy / elasticModulusSteel, steelStrains, curvatures)
    fycurv = min(fycurvC, fycurvS)
    fyM = np.interp(fycurv, curvatures, moments)
    eqcurv = max((mn / fyM) * fycurv, fycurv)
    curvbilin = np.array([0.0, eqcurv, curvatures[-1]])
    mombilin = np.array([0.0, mn, moments[-1]])

    sectionCurvatureDuctility = curvatures[-1] / eqcurv
    dbl = np.max(barDiameters)
    lsp = []
    for ss in steelStrains:
        ffss = -ss * elasticModulusSteel
        ffss = min(ffss, fy)
        lsp.append(kLsp * ffss * dbl)
    lsp = np.array(lsp)
    kkk = min(0.2 * (fsu / fy - 1), 0.08)

    if bending.lower() == "single":
        lp = max(kkk * memberLength + kLsp * fy * dbl, 2 * kLsp * fy * dbl)
        lbe = memberLength
    elif bending.lower() == "double":
        lp = max(kkk * memberLength / 2 + kLsp * fy * dbl, 2 * kLsp * fy * dbl)
        lbe = memberLength / 2
    else:
        raise ValueError("bending should be 'single' or 'double'")

    cuDu = curvatures / eqcurv
    bucritMK = 0
    failCuDuMK = None
    esfl = None
    if sectionCurvatureDuctility > 4:
        esgr4 = -0.5 * np.interp(4, cuDu, steelStrains)
        escc = 3 * (transverseSpacing / barDiameters[0]) ** (-2.5)
        esgr = np.zeros_like(steelStrains)
        for i, duct in enumerate(cuDu):
            if duct < 1:
                esgr[i] = 0
            elif duct < 4:
                esgr[i] = (esgr4 / 4) * duct
            else:
                esgr[i] = -0.5 * steelStrains[i]
        esfl = escc - esgr
        if -steelStrains[-1] >= esfl[-1]:
            bucritMK = 1
            fail = esfl - (-steelStrains)
            failCuDuMK = np.interp(0, fail, cuDu)

    bucritBE = 0
    c0 = 0.019
    c1 = 1.650
    c2 = 1.797
    c3 = 0.012
    c4 = 0.072
    transvSteelRatioAvg = ((hoopsX * 0.25 * math.pi * hoopDiameter**2) / (transverseSpacing * heightCore) + (hoopsY * 0.25 * math.pi * hoopDiameter**2) / (transverseSpacing * widthCore)) * 0.5
    roeff = (2 * transvSteelRatioAvg) * fyh / fpc
    rotb = c0 * (1 + c1 * roeff) * (1 + c2 * axialLoad / (height * width * fpc)) ** (-1) * (1 + c3 * lbe / height + c4 * barDiameters[0] * fy / height)
    plrot = (curvatures - fycurv) * (lp) / 1000
    if np.max(plrot) > rotb:
        bucritBE = 1
        failBE = plrot - rotb
        failCuDuBE = np.interp(0, failBE, cuDu)

    gShear = 0.43 * ecModulus
    asArea = (5 / 6) * height * width
    igross = width * height**3 / 12
    ieff = (mn * 1000 / (ecModulus * 1e6 * eqcurv)) * 1e12
    beta = min(0.5 + 20 * (ast / (height * width)), 1)
    if bending.lower() == "single":
        alpha = min(max(1, 3 - memberLength / height), 1.5)
        forceVector = moments / (memberLength / 1000)
    else:
        alpha = min(max(1, 3 - memberLength / (2 * height)), 1.5)
        forceVector = 2 * moments / (memberLength / 1000)
    vc1 = 0.29 * alpha * beta * 0.8 * math.sqrt(fpc) * height * width / 1000
    kscr = (0.25 * (hoopsY * 0.25 * math.pi * hoopDiameter**2 / (transverseSpacing * widthCore)) * elasticModulusSteel * (width / 1000) * ((height - dcore) / 1000) / (0.25 + 10 * (hoopsY * 0.25 * math.pi * hoopDiameter**2 / (transverseSpacing * widthCore)))) * 1000
    if bending.lower() == "single":
        ksg = (gShear * asArea / memberLength) / 1000
        kscr /= memberLength
        forcebilin = mombilin / (memberLength / 1000)
    else:
        ksg = (gShear * asArea / (memberLength / 2)) / 1000
        kscr /= (memberLength / 2)
        forcebilin = 2 * mombilin / (memberLength / 1000)
    kseff = ksg * (ieff / igross)
    aux = (vc1 / kseff) / 1000
    displsh: List[float] = []
    aux2 = 0
    momaux = moments.copy()
    for i, cur in enumerate(curvatures):
        if momaux[i] <= mn and forceVector[i] < vc1:
            displsh.append((forceVector[i] / kseff) / 1000)
        elif momaux[i] <= mn and forceVector[i] >= vc1:
            displsh.append(((forceVector[i] - vc1) / kscr) / 1000 + aux)
        else:
            aux2 += 1
            displsh.append((cur / curvatures[i - 1]) * displsh[i - 1])
    displsh = np.array(displsh)

    displf = []
    for idx, cur in enumerate(curvatures):
        if coverStrains[idx] < (0.56 * math.sqrt(fpc)) / ecModulus:
            displf.append(cur * ((memberLength / 1000) ** 2) / (3 if bending.lower() == "single" else 6))
        elif coverStrains[idx] > (0.56 * math.sqrt(fpc)) / ecModulus and cur < fycurv:
            factor = (memberLength + (lsp[idx] if bending.lower() == "single" else 2 * lsp[idx])) / 1000
            displf.append(cur * (factor**2) / (3 if bending.lower() == "single" else 6))
        else:
            factor = (memberLength + (lsp[idx] if bending.lower() == "single" else 2 * lsp[idx]) - 0.5 * lp) / 1000
            factor2 = (memberLength + (lsp[idx] if bending.lower() == "single" else 2 * lsp[idx])) / 1000
            displf.append((cur - fycurv * (moments[idx] / fyM)) * (lp / 1000) * factor + (fycurv * (factor2**2) / (3 if bending.lower() == "single" else 6)) * (moments[idx] / fyM))
    displf = np.array(displf)
    displ = displsh + displf
    dy1 = np.interp(fycurv, curvatures, displ)
    dy = (mn / fyM) * dy1
    du = displ[-1]
    displbilin = np.array([0.0, dy, du])
    ductility = displ / dy
    displDuct = np.max(ductility)
    dy1f = np.interp(fycurv, curvatures, displf)
    dyf = (mn / fyM) * dy1f
    vs = (hoopsY * 0.25 * math.pi * hoopDiameter**2 * fyh * (height - coverToLongBars + 0.5 * hoopDiameter - cMn) * 1 / math.tan(math.pi / 6) / transverseSpacing) / 1000
    vsd = (hoopsY * 0.25 * math.pi * hoopDiameter**2 * fyh * (height - coverToLongBars + 0.5 * hoopDiameter - cMn) * 1 / math.tan(math.radians(35)) / transverseSpacing) / 1000
    betaShear = min(0.5 + 20 * (ast / (height * width)), 1)
    ductilityFlex = displ / dyf
    if bending.lower() == "single":
        alphaShear = min(max(1, 3 - memberLength / height), 1.5)
        vp = (axialLoad * (height - cMn) / (2 * memberLength)) / 1000 if axialLoad > 0 else 0
    else:
        alphaShear = min(max(1, 3 - memberLength / (2 * height)), 1.5)
        vp = (axialLoad * (height - cMn) / (memberLength)) / 1000 if axialLoad > 0 else 0
    vc = []
    for ddf in ductilityFlex:
        if ductilityMode.lower() == "uniaxial":
            vc.append(alphaShear * betaShear * min(max(0.05, 0.37 - 0.04 * ddf), 0.29) * 0.8 * math.sqrt(fpc) * height * width / 1000)
        else:
            vc.append(alphaShear * betaShear * min(max(0.05, 0.33 - 0.04 * ddf), 0.29) * 0.8 * math.sqrt(fpc) * height * width / 1000)
    vc = np.array(vc)
    vcd = 0.862 * vc
    vpd = 0.85 * vp
    v = vc + vs + vp
    vd = 0.85 * (vcd + vsd + vpd)
    criteria = 1
    faildispl = failforce = failduct = failmom = failcurv = failCuDu = None
    if v[-1] < forceVector[-1]:
        failure = v - forceVector
        faildispl = np.interp(0, failure, displ)
        failforce = np.interp(faildispl, displ, forceVector)
        failduct = np.interp(faildispl, displ, ductility)
        failmom = np.interp(faildispl, displ, moments)
        failcurv = np.interp(faildispl, displ, curvatures)
        failCuDu = np.interp(faildispl, displ, cuDu)
        if bending.lower() == "single":
            if faildispl <= 2 * dy:
                criteria = 2
            elif faildispl < 8 * dy:
                criteria = 3
            else:
                criteria = 4
        else:
            if faildispl <= 1 * dy:
                criteria = 2
            elif faildispl < 7 * dy:
                criteria = 3
            else:
                criteria = 4

    ieq = (mn / (eqcurv * ecModulus)) / 1000
    bi = 1 / ((mombilin[1] / curvbilin[1]) / ((mombilin[2] - mombilin[1]) / (curvbilin[2] - curvbilin[1])))

    displdam = displser = dductdam = dductser = curvdam = curvser = cududam = cuduser = coverstraindam = coverstrainser = steelstraindam = steelstrainser = momdam = momser = forcedam = forceser = 0.0
    if np.max(coverStrains) > ecser or np.max(np.abs(steelStrains)) > abs(esser):
        if np.max(coverStrains) > ecdam or np.max(np.abs(steelStrains)) > abs(esdam):
            displdamc = np.interp(ecdam, coverStrains, displ)
            displdams = np.interp(esdam, steelStrains, displ)
            displdam = min(displdamc, displdams)
            dductdam = np.interp(displdam, displ, ductility)
            curvdam = np.interp(displdam, displ, curvatures)
            cududam = np.interp(displdam, displ, cuDu)
            coverstraindam = np.interp(displdam, displ, coverStrains)
            steelstraindam = np.interp(displdam, displ, steelStrains)
            momdam = np.interp(displdam, displ, moments)
            forcedam = np.interp(displdam, displ, forceVector)

        displserc = np.interp(ecser, coverStrains, displ)
        displsers = np.interp(esser, steelStrains, displ)
        displser = min(displserc, displsers)
        dductser = np.interp(displser, displ, ductility)
        curvser = np.interp(displser, displ, curvatures)
        cuduser = np.interp(displser, displ, cuDu)
        coverstrainser = np.interp(displser, displ, coverStrains)
        steelstrainser = np.interp(displser, displ, steelStrains)
        momser = np.interp(displser, displ, moments)
        forceser = np.interp(displser, displ, forceVector)

    _ = (cuDu, esfl, failCuDuMK, failCuDu)
    plotStressStrain(concreteConfined, concreteUnconfined, steelModel)
    plotMomentCurvature(curvbilin, mombilin, curvatures, moments, "Moment - Curvature Relation (Rectangular)")
    plotBucklingModels(cuDu, steelStrains, esfl, failCuDuMK, bucritMK, plrot, rotb, bucritBE, "rectangular")
    plotForceDisplacement(displbilin, forcebilin, displ, forceVector, v, vd, criteria, faildispl, failforce, bucritMK, cuDu, failCuDuMK, bucritBE, failCuDuBE, "rectangular")

    results = AnalysisResults(
        moments=moments,
        curvatures=curvatures,
        neutralAxes=neutralAxes,
        coverStrains=coverStrains,
        coreStrains=coreStrains,
        steelStrains=steelStrains,
        forces=forceVector,
        shearDisplacements=displsh,
        flexuralDisplacements=displf,
        displacements=displ,
        shearCapacityAssess=v,
        shearCapacityDesign=vd,
        bilinearCurvatures=curvbilin,
        bilinearMoments=mombilin,
        bilinearDisplacements=displbilin,
        bilinearForces=forcebilin,
        additional={
            "nominalMoment": mn,
            "eqCurvature": eqcurv,
            "sectionCurvatureDuctility": sectionCurvatureDuctility,
            "displacementDuctility": displDuct,
            "yieldCurvature": fycurv,
            "yieldMoment": fyM,
            "plasticHingeLength": lp,
            "biFactor": bi,
            "equivalentInertia": ieq,
            "criteria": criteria,
            "bucklingMK": bucritMK,
            "bucklingBE": bucritBE,
        },
    )
    writeOutputFileRectangular(name, config, results, conclay, concreteConfined, concreteUnconfined, steelModel, mn, eqcurv, sectionCurvatureDuctility, displDuct, criteria, faildispl, failforce, failcurv, failCuDu, failmom, cuDu, ductility, bucritMK, bucritBE, failCuDuMK if failCuDuMK else 0, failCuDuBE if bucritBE else 0)
    return results


# --------------------------------------------------------------------------- #
# Circular analysis
# --------------------------------------------------------------------------- #


def analyzeCircularSection(config: Dict) -> AnalysisResults:
    """
    Execute the circular CUMBIA workflow.
    """

    name = config["name"]
    interaction = config["interaction"]
    diameter = config["diameter"]
    coverToLongBars = config["coverToLongBars"]
    memberLength = config["memberLength"]
    bending = config["bending"]
    ductilityMode = config["ductilityMode"]
    nbLong = config["nbLong"]
    longDiameter = config["longDiameter"]
    hoopDiameter = config["hoopDiameter"]
    hoopType = config["hoopType"]
    transverseSpacing = config["transverseSpacing"]
    axialLoad = config["axialLoad"] * 1000
    confinedModelName = config["confinedModel"]
    unconfinedModelName = config["unconfinedModel"]
    steelModelName = config["steelModel"]
    fpc = config["fpc"]
    ecModulus = 5000 * math.sqrt(fpc) if config["ecModulus"] == 0 else config["ecModulus"]
    eco = config["eco"]
    esm = config["esm"]
    espAll = config["espAll"]
    fy = config["fy"]
    fyh = config["fyh"]
    elasticModulusSteel = config["elasticModulusSteel"]
    fsu = config["fsu"]
    esh = config["esh"]
    esu = config["esu"]
    eySlope = config["eySlope"]
    c1Value = config["c1Value"]
    csid = config["csid"]
    ssid = config["ssid"]
    ecser = config["ecser"]
    esser = -config["esser"]
    ecdam = config["ecdam"]
    esdam = -config["esdam"]
    temp = config["temperature"]
    kLsp = config["kLsp"]
    iterMax = config["iterMax"]
    ncl = config["ncl"]
    tolerance = config["tolerance"]
    strainStep = config["strainStep"]

    coreDiameter = diameter - 2 * coverToLongBars + hoopDiameter
    dcore = coverToLongBars - hoopDiameter * 0.5
    ast = nbLong * 0.25 * math.pi * longDiameter**2
    conclay = None
    if temp < 0:
        ct = (1 - 0.0105 * temp) * 0.56 * math.sqrt(fpc)
    else:
        ct = 0.56 * math.sqrt(fpc)
    eccr = ct / ecModulus

    concreteUnconfined = buildConcreteModel(unconfinedModelName, ecModulus, ast, hoopDiameter, coverToLongBars, transverseSpacing, fpc, fyh, eco, esm, espAll, "circular", diameter, 0, 0, 0, 0, [], strainStep, hoopType)
    concreteConfined = buildConcreteModel(confinedModelName, ecModulus, ast, hoopDiameter, coverToLongBars, transverseSpacing, fpc, fyh, eco, esm, espAll, "circular", diameter, 0, 0, 0, 0, [], strainStep, hoopType)
    steelModel = buildSteelModel(steelModelName, elasticModulusSteel, fy, fsu, esh, esu, strainStep, c1Value, eySlope)
    esuLimit = steelModel[0][-2]
    if isinstance(ecdam, str) and ecdam.lower() == "twth":
        ecdam = (concreteConfined[0][-2]) / 1.5

    theta = (2 * math.pi / nbLong) * np.arange(nbLong)
    r = 0.5 * (diameter - 2 * coverToLongBars - longDiameter)
    distld = np.sort(0.5 * (diameter - 2 * r) + r * np.sin(theta) * np.tan(0.5 * theta))
    asbs = np.full(nbLong, 0.25 * math.pi * longDiameter**2)
    conclay = circularLayerAreas(diameter, coreDiameter, coverToLongBars, hoopDiameter, ncl, asbs, distld)
    defVector = buildDeflectionVector(concreteConfined[0][-2], strainStep)

    if axialLoad > 0:
        for idx in range(len(defVector)):
            compCheck = np.sum(interpolation(concreteUnconfined[1], concreteUnconfined[0], defVector[0] * np.ones(len(conclay))) * conclay[:, 1]) + np.sum(interpolation(concreteConfined[1], concreteConfined[0], defVector[0] * np.ones(len(conclay))) * conclay[:, 2]) + np.sum(asbs * interpolation(steelModel[1], steelModel[0], defVector[0]))
            if compCheck < axialLoad:
                defVector = defVector[1:]
            else:
                break

    moments, curvatures, neutralAxes, coverStrains, coreStrains, steelStrains, messageFlag = computeMomentCurvature(
        "circular",
        defVector,
        tolerance * 0.25 * math.pi * diameter**2 * fpc,
        iterMax,
        axialLoad,
        conclay,
        distld,
        asbs,
        concreteUnconfined,
        concreteConfined,
        steelModel,
        esuLimit,
        concreteConfined[0][-2],
        diameter,
    )

    mn = np.interp(ecser, coverStrains, moments)
    esaux = np.interp(ecser, coverStrains, steelStrains)
    cr = 0
    if abs(esaux) > abs(esser) or math.isnan(mn):
        mnSteel = np.interp(esser, steelStrains, moments)
        if not math.isnan(mnSteel):
            cr = 1
            mn = mnSteel
        else:
            raise RuntimeError("Problem estimating nominal moment")

    cMn = np.interp(mn, moments, neutralAxes)
    fycurvC = np.interp(1.8 * fpc / ecModulus, coverStrains, curvatures)
    fycurvS = np.interp(-fy / elasticModulusSteel, steelStrains, curvatures)
    fycurv = min(fycurvC, fycurvS)
    fyM = np.interp(fycurv, curvatures, moments)
    eqcurv = max((mn / fyM) * fycurv, fycurv)
    curvbilin = np.array([0.0, eqcurv, curvatures[-1]])
    mombilin = np.array([0.0, mn, moments[-1]])
    sectionCurvatureDuctility = curvatures[-1] / eqcurv

    lsp = []
    for ss in steelStrains:
        ffss = -ss * elasticModulusSteel
        ffss = min(ffss, fy)
        lsp.append(kLsp * ffss * longDiameter)
    lsp = np.array(lsp)
    kkk = min(0.2 * (fsu / fy - 1), 0.08)

    if bending.lower() == "single":
        lp = max(kkk * memberLength + kLsp * fy * longDiameter, 2 * kLsp * fy * longDiameter)
        lbe = memberLength
        forceVector = moments / (memberLength / 1000)
    else:
        lp = max(kkk * memberLength / 2 + kLsp * fy * longDiameter, 2 * kLsp * fy * longDiameter)
        lbe = memberLength / 2
        forceVector = 2 * moments / (memberLength / 1000)

    cuDu = curvatures / eqcurv
    bucritMK = 0
    esfl = None
    failCuDuMK = None
    if sectionCurvatureDuctility > 4:
        esgr4 = -0.5 * np.interp(4, cuDu, steelStrains)
        escc = 3 * (transverseSpacing / longDiameter) ** (-2.5)
        esgr = np.zeros_like(steelStrains)
        for i, duct in enumerate(cuDu):
            if duct < 1:
                esgr[i] = 0
            elif duct < 4:
                esgr[i] = (esgr4 / 4) * duct
            else:
                esgr[i] = -0.5 * steelStrains[i]
        esfl = escc - esgr
        if -steelStrains[-1] >= esfl[-1]:
            bucritMK = 1
            fail = esfl - (-steelStrains)
            failCuDuMK = np.interp(0, fail, cuDu)

    bucritBE = 0
    if (axialLoad / (fpc * (0.25 * math.pi * diameter**2))) >= 0.30:
        c0 = 0.006
        c1 = 7.190
        c2 = 3.129
        c3 = 0.651
        c4 = 0.227
    else:
        c0 = 0.001
        c1 = 7.30
        c2 = 1.30
        c3 = 1.30
        c4 = 3.00
    transvSteelRatio = math.pi * hoopDiameter**2 / (transverseSpacing * coreDiameter)
    roeff = transvSteelRatio * fyh / fpc
    rotb = c0 * (1 + c1 * roeff) * ((1 + c2 * axialLoad / (0.25 * math.pi * diameter**2 * fpc)) ** (-1)) * (1 + c3 * lbe / diameter + c4 * longDiameter * fy / diameter)
    plrot = (curvatures - fycurv) * lp / 1000
    if np.max(plrot) > rotb:
        bucritBE = 1
        failBE = plrot - rotb
        failCuDuBE = np.interp(0, failBE, cuDu)
    else:
        failCuDuBE = None

    gShear = 0.43 * ecModulus
    asArea = 0.9 * (0.25 * math.pi * diameter**2)
    igross = math.pi * diameter**4 / 64
    ieff = (mn * 1000 / (ecModulus * 1e6 * eqcurv)) * 1e12
    beta = min(0.5 + 20 * (ast / (0.25 * math.pi * diameter**2)), 1)
    alpha = min(max(1, 3 - memberLength / (diameter if bending.lower() == "single" else 2 * diameter)), 1.5)
    vc1 = 0.29 * alpha * beta * 0.8 * math.sqrt(fpc) * (0.25 * math.pi * diameter**2) / 1000
    kscr = ((0.39 * transvSteelRatio) * 0.25 * elasticModulusSteel * ((0.8 * diameter / 1000) ** 2) / (0.25 + 10 * (0.39 * transvSteelRatio))) * 1000
    if bending.lower() == "single":
        ksg = (gShear * asArea / memberLength) / 1000
        kscr /= memberLength
        forcebilin = mombilin / (memberLength / 1000)
    else:
        ksg = (gShear * asArea / (memberLength / 2)) / 1000
        kscr /= (memberLength / 2)
        forcebilin = 2 * mombilin / (memberLength / 1000)
    kseff = ksg * (ieff / igross)
    aux = (vc1 / kseff) / 1000
    displsh: List[float] = []
    aux2 = 0
    momaux = moments.copy()
    for i, cur in enumerate(curvatures):
        if momaux[i] <= mn and forceVector[i] < vc1:
            displsh.append((forceVector[i] / kseff) / 1000)
        elif momaux[i] <= mn and forceVector[i] >= vc1:
            displsh.append(((forceVector[i] - vc1) / kscr) / 1000 + aux)
        else:
            aux2 += 1
            displsh.append((cur / curvatures[i - 1]) * displsh[i - 1])
    displsh = np.array(displsh)

    displf = []
    for idx, cur in enumerate(curvatures):
        if coverStrains[idx] < eccr:
            displf.append(cur * ((memberLength / 1000) ** 2) / (3 if bending.lower() == "single" else 6))
        elif coverStrains[idx] > eccr and cur < fycurv:
            factor = (memberLength + (lsp[idx] if bending.lower() == "single" else 2 * lsp[idx])) / 1000
            displf.append(cur * (factor**2) / (3 if bending.lower() == "single" else 6))
        else:
            factor = (memberLength + (lsp[idx] if bending.lower() == "single" else 2 * lsp[idx]) - 0.5 * lp) / 1000
            factor2 = (memberLength + (lsp[idx] if bending.lower() == "single" else 2 * lsp[idx])) / 1000
            displf.append((cur - fycurv * (moments[idx] / fyM)) * (lp / 1000) * factor + (fycurv * (factor2**2) / (3 if bending.lower() == "single" else 6)) * (moments[idx] / fyM))
    displf = np.array(displf)
    displ = displsh + displf
    dy1 = np.interp(fycurv, curvatures, displ)
    dy = (mn / fyM) * dy1
    du = displ[-1]
    displbilin = np.array([0.0, dy, du])
    ductility = displ / dy
    displDuct = np.max(ductility)
    dy1f = np.interp(fycurv, curvatures, displf)
    dyf = (mn / fyM) * dy1f
    vs = (0.5 * math.pi * (0.25 * math.pi * hoopDiameter**2) * fyh * 1 / math.tan(math.pi / 6) * (diameter - coverToLongBars + 0.5 * hoopDiameter - cMn) / transverseSpacing) / 1000
    vsd = (0.5 * math.pi * (0.25 * math.pi * hoopDiameter**2) * fyh * 1 / math.tan(math.radians(35)) * (diameter - coverToLongBars + 0.5 * hoopDiameter - cMn) / transverseSpacing) / 1000
    betaShear = min(0.5 + 20 * (ast / (0.25 * math.pi * diameter**2)), 1)
    ductilityFlex = displ / dyf
    if bending.lower() == "single":
        alphaShear = min(max(1, 3 - memberLength / diameter), 1.5)
        vp = (axialLoad * (diameter - cMn) / (2 * memberLength)) / 1000 if axialLoad > 0 else 0
    else:
        alphaShear = min(max(1, 3 - memberLength / (2 * diameter)), 1.5)
        vp = (axialLoad * (diameter - cMn) / (memberLength)) / 1000 if axialLoad > 0 else 0
    vc = []
    for ddf in ductilityFlex:
        if ductilityMode.lower() == "uniaxial":
            vc.append(alphaShear * betaShear * min(max(0.05, 0.37 - 0.04 * ddf), 0.29) * 0.8 * math.sqrt(fpc) * (0.25 * math.pi * diameter**2) / 1000)
        else:
            vc.append(alphaShear * betaShear * min(max(0.05, 0.33 - 0.04 * ddf), 0.29) * 0.8 * math.sqrt(fpc) * (0.25 * math.pi * diameter**2) / 1000)
    vc = np.array(vc)
    vcd = 0.862 * vc
    vpd = 0.85 * vp
    v = vc + vs + vp
    vd = 0.85 * (vcd + vsd + vpd)
    criteria = 1
    faildispl = failforce = failduct = failmom = failcurv = failCuDu = None
    if v[-1] < forceVector[-1]:
        failure = v - forceVector
        faildispl = np.interp(0, failure, displ)
        failforce = np.interp(faildispl, displ, forceVector)
        failduct = np.interp(faildispl, displ, ductility)
        failmom = np.interp(faildispl, displ, moments)
        failcurv = np.interp(faildispl, displ, curvatures)
        failCuDu = np.interp(faildispl, displ, cuDu)
        if bending.lower() == "single":
            if faildispl <= 2 * dy:
                criteria = 2
            elif faildispl < 8 * dy:
                criteria = 3
            else:
                criteria = 4
        else:
            if faildispl <= 1 * dy:
                criteria = 2
            elif faildispl < 7 * dy:
                criteria = 3
            else:
                criteria = 4

    ieq = (mn / (eqcurv * ecModulus)) / 1000
    bi = 1 / ((mombilin[1] / curvbilin[1]) / ((mombilin[2] - mombilin[1]) / (curvbilin[2] - curvbilin[1])))

    displdam = displser = dductdam = dductser = curvdam = curvser = cududam = cuduser = coverstraindam = coverstrainser = steelstraindam = steelstrainser = momdam = momser = forcedam = forceser = 0.0
    if np.max(coverStrains) > ecser or np.max(np.abs(steelStrains)) > abs(esser):
        if np.max(coverStrains) > ecdam or np.max(np.abs(steelStrains)) > abs(esdam):
            displdamc = np.interp(ecdam, coverStrains, displ)
            displdams = np.interp(esdam, steelStrains, displ)
            displdam = min(displdamc, displdams)
            dductdam = np.interp(displdam, displ, ductility)
            curvdam = np.interp(displdam, displ, curvatures)
            cududam = np.interp(displdam, displ, cuDu)
            coverstraindam = np.interp(displdam, displ, coverStrains)
            steelstraindam = np.interp(displdam, displ, steelStrains)
            momdam = np.interp(displdam, displ, moments)
            forcedam = np.interp(displdam, displ, forceVector)
        displserc = np.interp(ecser, coverStrains, displ)
        displsers = np.interp(esser, steelStrains, displ)
        displser = min(displserc, displsers)
        dductser = np.interp(displser, displ, ductility)
        curvser = np.interp(displser, displ, curvatures)
        cuduser = np.interp(displser, displ, cuDu)
        coverstrainser = np.interp(displser, displ, coverStrains)
        steelstrainser = np.interp(displser, displ, steelStrains)
        momser = np.interp(displser, displ, moments)
        forceser = np.interp(displser, displ, forceVector)

    plotStressStrain(concreteConfined, concreteUnconfined, steelModel)
    plotMomentCurvature(curvbilin, mombilin, curvatures, moments, "Moment - Curvature Relation (Circular)")
    plotBucklingModels(cuDu, steelStrains, esfl, failCuDuMK, bucritMK, plrot, rotb, bucritBE, "circular")
    plotForceDisplacement(displbilin, forcebilin, displ, forceVector, v, vd, criteria, faildispl, failforce, bucritMK, cuDu, failCuDuMK, bucritBE, failCuDuBE, "circular")

    results = AnalysisResults(
        moments=moments,
        curvatures=curvatures,
        neutralAxes=neutralAxes,
        coverStrains=coverStrains,
        coreStrains=coreStrains,
        steelStrains=steelStrains,
        forces=forceVector,
        shearDisplacements=displsh,
        flexuralDisplacements=displf,
        displacements=displ,
        shearCapacityAssess=v,
        shearCapacityDesign=vd,
        bilinearCurvatures=curvbilin,
        bilinearMoments=mombilin,
        bilinearDisplacements=displbilin,
        bilinearForces=forcebilin,
        additional={
            "nominalMoment": mn,
            "eqCurvature": eqcurv,
            "sectionCurvatureDuctility": sectionCurvatureDuctility,
            "displacementDuctility": displDuct,
            "yieldCurvature": fycurv,
            "yieldMoment": fyM,
            "plasticHingeLength": lp,
            "biFactor": bi,
            "equivalentInertia": ieq,
            "criteria": criteria,
            "bucklingMK": bucritMK,
            "bucklingBE": bucritBE,
        },
    )
    writeOutputFileCircular(name, config, results, conclay, concreteConfined, concreteUnconfined, steelModel, mn, eqcurv, sectionCurvatureDuctility, displDuct, criteria, faildispl, failforce, failcurv, failCuDu, failmom, cuDu, ductility, bucritMK, bucritBE, failCuDuMK if failCuDuMK else 0, failCuDuBE if bucritBE else 0)
    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plotStressStrain(concreteConfined, concreteUnconfined, steelModel) -> None:
    """
    Plot concrete and steel stress-strain curves.
    """

    ec, fc = concreteConfined
    ecun, fcun = concreteUnconfined
    es, fs = steelModel
    plt.figure()
    plt.fill_between(ec, fc, color="c", label="Confined Concrete")
    plt.fill_between(ecun, fcun, color="b", alpha=0.5, label="Unconfined Concrete")
    plt.xlabel("Strain")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    plt.title("Stress-Strain Relation for Concrete")
    plt.grid(True)

    plt.figure()
    plt.fill_between(es, fs, color=(0.8, 0.8, 0.4))
    plt.xlabel("Strain")
    plt.ylabel("Stress [MPa]")
    plt.title("Stress-Strain Relation for Reinforcing Steel")
    plt.grid(True)


def plotMomentCurvature(curvbilin, mombilin, curvatures, moments, title: str) -> None:
    """
    Plot the moment-curvature response with bilinear approximation.
    """

    plt.figure()
    plt.plot(curvbilin, mombilin, "r", linewidth=2, label="Bilinear")
    plt.plot(curvatures, moments, "b--", linewidth=2, label="Full")
    plt.xlabel("Curvature (1/m)")
    plt.ylabel("Moment (kN-m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()


def plotBucklingModels(cuDu, steelStrains, esfl, failCuDuMK, bucritMK, plrot, rotb, bucritBE, label: str) -> None:
    """
    Plot buckling assessments (Moyer-Kowalsky and Berry-Eberhard).
    """

    if esfl is not None:
        plt.figure()
        plt.plot(cuDu, -steelStrains, "-r", label="Column strain ductility behavior")
        plt.plot(cuDu, esfl, "--b", label="Flexural Tension Strain")
        if bucritMK and failCuDuMK is not None:
            failss = -np.interp(failCuDuMK, cuDu, steelStrains)
            plt.plot(failCuDuMK, failss, "mo", markeredgecolor="k", markerfacecolor="g", markersize=10, label="Buckling")
        plt.xlabel("Curvature Ductility")
        plt.ylabel("Steel Tension Strain")
        plt.title(f"Moyer - Kowalsky Buckling Model ({label})")
        plt.legend()
        plt.grid(True)

    plt.figure()
    plt.plot(cuDu, rotb * np.ones_like(cuDu), "--r", label="Plastic Rotation for Buckling")
    plt.plot(cuDu, plrot, "-b", label="Plastic Rotation")
    if bucritBE:
        failCuDuBE = np.interp(rotb, plrot, cuDu)
        plt.plot(failCuDuBE, rotb, "mo", markeredgecolor="k", markerfacecolor="g", markersize=10, label="Buckling")
    plt.xlabel("Curvature Ductility")
    plt.ylabel("Plastic Rotation")
    plt.title(f"Berry - Eberhard Buckling Model ({label})")
    plt.legend()
    plt.grid(True)


def plotForceDisplacement(displbilin, forcebilin, displ, forceVector, v, vd, criteria, faildispl, failforce, bucritMK, cuDu, failCuDuMK, bucritBE, failCuDuBE, label: str) -> None:
    """
    Plot the force-displacement response with shear capacities.
    """

    plt.figure()
    plt.plot(displbilin, forcebilin, "b--", linewidth=2, label="Bilinear approximation")
    plt.plot(displ, forceVector, "k", linewidth=2, label="Total response")
    plt.plot(displ, v, "r:", linewidth=2, label="Shear capacity (assessment)")
    plt.plot(displ, vd, "m:", linewidth=2, label="Shear capacity (design)")
    if criteria != 1 and faildispl is not None and failforce is not None:
        plt.plot(faildispl, failforce, "mo", markeredgecolor="k", markerfacecolor="g", markersize=10, label="Shear failure")
    if bucritMK and failCuDuMK is not None:
        bucklDispl = np.interp(failCuDuMK, cuDu, displ)
        bucklForce = np.interp(failCuDuMK, cuDu, forceVector)
        plt.plot(bucklDispl, bucklForce, "k*", markeredgecolor="k", markersize=10, markerfacecolor="g", label="Buckling (M & K)")
    if bucritBE and failCuDuBE is not None:
        bucklDisplBE = np.interp(failCuDuBE, cuDu, displ)
        bucklForceBE = np.interp(failCuDuBE, cuDu, forceVector)
        plt.plot(bucklDisplBE, bucklForceBE, "ks", markeredgecolor="k", markersize=10, markerfacecolor="g", label="Buckling (B & E)")
    plt.xlabel("Displacement (m)")
    plt.ylabel("Force (kN)")
    plt.title(f"Force - Displacement Relation ({label})")
    plt.legend()
    plt.grid(True)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def writeOutputFileRectangular(
    name: str,
    config: Dict,
    results: AnalysisResults,
    conclay: np.ndarray,
    concreteConfined: Tuple[np.ndarray, np.ndarray],
    concreteUnconfined: Tuple[np.ndarray, np.ndarray],
    steelModel: Tuple[np.ndarray, np.ndarray],
    mn: float,
    eqcurv: float,
    sectionCurvatureDuctility: float,
    displDuct: float,
    criteria: int,
    faildispl: float,
    failforce: float,
    failcurv: float,
    failCuDu: float,
    failmom: float,
    cuDu: np.ndarray,
    ductility: np.ndarray,
    bucritMK: int,
    bucritBE: int,
    failCuDuMK: float,
    failCuDuBE: float,
) -> None:
    """
    Write a human-readable text summary equivalent to the MATLAB .xls output.
    """

    outputPath = Path(f"{name}.txt")
    with outputPath.open("w") as fid:
        fid.write("Rectangular Section\n\n")
        fid.write(f"Width: {config['width']:.1f} mm   Height: {config['height']:.1f} mm\n")
        fid.write(f"cover to longitudinal bars: {config['coverToLongBars']:.1f} mm\n")
        fid.write("\nDist.Top\t# Long\tDiameter\n[mm]\tBars\t[mm]\n")
        for row in config["longitudinalRows"]:
            fid.write(f"{row[0]:.1f}\t{row[1]:.1f}\t{row[2]:.2f}\n")
        fid.write(f"\ndiameter of transverse steel: {config['hoopDiameter']:.1f} mm\n")
        fid.write(f"spacing of transverse steel: {config['transverseSpacing']:.1f} mm\n")
        fid.write(f"# legs transv. steel x_dir (confinement): {config['hoopsX']:.1f}\n")
        fid.write(f"# legs transv. steel y_dir (shear): {config['hoopsY']:.1f}\n")
        fid.write(f"axial load: {config['axialLoad']:.2f} kN\n")
        fid.write(f"concrete compressive strength: {config['fpc']:.2f} MPa\n")
        fid.write(f"long steel yielding stress: {config['fy']:.2f} MPa\n")
        fid.write(f"transverse steel yielding stress: {config['fyh']:.2f} MPa\n")
        fid.write(f"Member Length: {config['memberLength']:.1f} mm\n")
        fid.write(f"Longitudinal Steel Ratio: {config['longSteelRatio']:.3f}\n")
        fid.write(f"Average Transverse Steel Ratio: {config['transvSteelRatioAverage']:.3f}\n")
        fid.write(f"Axial Load Ratio: {config['axialRatio']:.3f}\n\n")
        fid.write("Cover\tCore\tN.A\tSteel\tMoment\tCurvature\tForce\tSh displ.\tFl displ.\tTotal displ.\tShear(assess.)\tShear(design)\n")
        fid.write("Strain\tStrain\t[mm]\tStrain\t[kN-m]\t[1/m]\t[kN]\t[m]\t[m]\t[m]\t[kN]\t[kN]\n")
        for idx in range(len(results.moments)):
            fid.write(
                f"{results.coverStrains[idx]:.5f}\t{results.coreStrains[idx]:.5f}\t{results.neutralAxes[idx]:.2f}\t{results.steelStrains[idx]:.5f}\t{results.moments[idx]:.2f}\t{results.curvatures[idx]:.5f}\t{results.forces[idx]:.2f}\t{results.shearDisplacements[idx]:.5f}\t{results.flexuralDisplacements[idx]:.5f}\t{results.displacements[idx]:.5f}\t{results.shearCapacityAssess[idx]:.2f}\t{results.shearCapacityDesign[idx]:.2f}\n"
            )
        fid.write("\nBilinear Approximation:\n\n")
        fid.write("Curvature\tMoment\tDispl.\tForce\n[1/m]\t[kN-m]\t[m]\t[kN]\n")
        for idx in range(len(results.bilinearCurvatures)):
            fid.write(f"{results.bilinearCurvatures[idx]:.5f}\t{results.bilinearMoments[idx]:.2f}\t{results.bilinearDisplacements[idx]:.5f}\t{results.bilinearForces[idx]:.2f}\n")
        if bucritMK:
            fid.write("Moyer - Kowalsky buckling predicted.\n")
        if bucritBE:
            fid.write("Berry - Eberhard buckling predicted.\n")
    return


def writeOutputFileCircular(
    name: str,
    config: Dict,
    results: AnalysisResults,
    conclay: np.ndarray,
    concreteConfined: Tuple[np.ndarray, np.ndarray],
    concreteUnconfined: Tuple[np.ndarray, np.ndarray],
    steelModel: Tuple[np.ndarray, np.ndarray],
    mn: float,
    eqcurv: float,
    sectionCurvatureDuctility: float,
    displDuct: float,
    criteria: int,
    faildispl: float,
    failforce: float,
    failcurv: float,
    failCuDu: float,
    failmom: float,
    cuDu: np.ndarray,
    ductility: np.ndarray,
    bucritMK: int,
    bucritBE: int,
    failCuDuMK: float,
    failCuDuBE: float,
) -> None:
    """
    Write the circular section summary.
    """

    outputPath = Path(f"{name}.txt")
    with outputPath.open("w") as fid:
        fid.write("Circular Section\n\n")
        fid.write(f"Diameter: {config['diameter']:.1f} mm\n")
        fid.write(f"cover to longitudinal bars: {config['coverToLongBars']:.1f} mm\n")
        fid.write(f"number of longitudinal bars: {config['nbLong']}\n")
        fid.write(f"longitudinal bar diameter: {config['longDiameter']:.1f} mm\n")
        fid.write(f"diameter of transverse steel: {config['hoopDiameter']:.1f} mm\n")
        fid.write(f"spacing of transverse steel: {config['transverseSpacing']:.1f} mm\n")
        fid.write(f"axial load: {config['axialLoad']:.2f} kN\n")
        fid.write(f"concrete compressive strength: {config['fpc']:.2f} MPa\n")
        fid.write(f"long steel yielding stress: {config['fy']:.2f} MPa\n")
        fid.write(f"transverse steel yielding stress: {config['fyh']:.2f} MPa\n")
        fid.write(f"Member Length: {config['memberLength']:.1f} mm\n")
        fid.write(f"Longitudinal Steel Ratio: {config['longSteelRatio']:.3f}\n")
        fid.write(f"Transverse Steel Ratio: {config['transvSteelRatio']:.3f}\n")
        fid.write(f"Axial Load Ratio: {config['axialRatio']:.3f}\n\n")
        fid.write("Cover\tCore\tN.A\tSteel\tMoment\tCurvature\tForce\tSh displ.\tFl displ.\tTotal displ.\tShear(assess.)\tShear(design)\n")
        fid.write("Strain\tStrain\t[mm]\tStrain\t[kN-m]\t[1/m]\t[kN]\t[m]\t[m]\t[m]\t[kN]\t[kN]\n")
        for idx in range(len(results.moments)):
            fid.write(
                f"{results.coverStrains[idx]:.5f}\t{results.coreStrains[idx]:.5f}\t{results.neutralAxes[idx]:.2f}\t{results.steelStrains[idx]:.5f}\t{results.moments[idx]:.2f}\t{results.curvatures[idx]:.5f}\t{results.forces[idx]:.2f}\t{results.shearDisplacements[idx]:.5f}\t{results.flexuralDisplacements[idx]:.5f}\t{results.displacements[idx]:.5f}\t{results.shearCapacityAssess[idx]:.2f}\t{results.shearCapacityDesign[idx]:.2f}\n"
            )
        fid.write("\nBilinear Approximation:\n\n")
        fid.write("Curvature\tMoment\tDispl.\tForce\n[1/m]\t[kN-m]\t[m]\t[kN]\n")
        for idx in range(len(results.bilinearCurvatures)):
            fid.write(f"{results.bilinearCurvatures[idx]:.5f}\t{results.bilinearMoments[idx]:.2f}\t{results.bilinearDisplacements[idx]:.5f}\t{results.bilinearForces[idx]:.2f}\n")
        if bucritMK:
            fid.write("Moyer - Kowalsky buckling predicted.\n")
        if bucritBE:
            fid.write("Berry - Eberhard buckling predicted.\n")
    return


# --------------------------------------------------------------------------- #
# Example configurations and entry points
# --------------------------------------------------------------------------- #


def defaultRectangularConfig() -> Dict:
    """
    Default rectangular configuration mirroring CUMBIARECT.m.
    """

    mlr = np.array([[52.7, 3, 25.4], [102, 2, 22.2], [200, 2, 19], [349, 3, 22.2]])
    agross = 400 * 300
    ast = np.sum(mlr[:, 1] * 0.25 * math.pi * mlr[:, 2] ** 2)
    longSteelRatio = ast / agross
    transvSteelRatioAverage = ((2 * 0.25 * math.pi * 9.5**2) / (120 * (400 - 2 * 40 + 9.5)) + (2 * 0.25 * math.pi * 9.5**2) / (120 * (300 - 2 * 40 + 9.5))) * 0.5
    axialRatio = (-200 * 1000) / (28 * agross)
    return {
        "name": "CUMBIARECTEX",
        "interaction": True,
        "height": 400.0,
        "width": 300.0,
        "hoopsX": 2,
        "hoopsY": 2,
        "coverToLongBars": 40.0,
        "memberLength": 1200.0,
        "bending": "single",
        "ductilityMode": "uniaxial",
        "longitudinalRows": mlr,
        "hoopDiameter": 9.5,
        "transverseSpacing": 120.0,
        "axialLoad": -200.0,
        "confinedModel": "mc",
        "unconfinedModel": "mu",
        "steelModel": "ra",
        "wiVector": [272, 272, 172, 172],
        "fpc": 28.0,
        "ecModulus": 0.0,
        "eco": 0.002,
        "esm": 0.12,
        "espAll": 0.0064,
        "fy": 450.0,
        "fyh": 400.0,
        "elasticModulusSteel": 200000.0,
        "fsu": 600.0,
        "esh": 0.008,
        "esu": 0.15,
        "eySlope": 350.0,
        "c1Value": 3.5,
        "csid": 0.004,
        "ssid": 0.015,
        "ecser": 0.004,
        "esser": 0.015,
        "ecdam": "twth",
        "esdam": 0.060,
        "temperature": 30.0,
        "kLsp": 0.022,
        "iterMax": 1000,
        "ncl": 40,
        "tolerance": 0.001,
        "strainStep": 0.0001,
        "longSteelRatio": longSteelRatio,
        "transvSteelRatioAverage": transvSteelRatioAverage,
        "axialRatio": axialRatio,
    }


def defaultCircularConfig() -> Dict:
    """
    Default circular configuration mirroring CUMBIACIR.m.
    """

    agross = 0.25 * math.pi * 1000**2
    ast = 22 * 0.25 * math.pi * 25**2
    longSteelRatio = ast / agross
    transvSteelRatio = math.pi * 9**2 / (120 * (1000 - 2 * 50 + 9))
    axialRatio = (400 * 1000) / (35 * agross)
    return {
        "name": "CUMBIACIREX",
        "interaction": True,
        "diameter": 1000.0,
        "coverToLongBars": 50.0,
        "memberLength": 3000.0,
        "bending": "single",
        "ductilityMode": "biaxial",
        "nbLong": 22,
        "longDiameter": 25.0,
        "hoopDiameter": 9.0,
        "hoopType": "spirals",
        "transverseSpacing": 120.0,
        "axialLoad": 400.0,
        "confinedModel": "mc",
        "unconfinedModel": "mu",
        "steelModel": "ra",
        "fpc": 35.0,
        "ecModulus": 0.0,
        "eco": 0.002,
        "esm": 0.11,
        "espAll": 0.0064,
        "fy": 460.0,
        "fyh": 400.0,
        "elasticModulusSteel": 200000.0,
        "fsu": 620.0,
        "esh": 0.008,
        "esu": 0.12,
        "eySlope": 350.0,
        "c1Value": 3.5,
        "csid": 0.004,
        "ssid": 0.015,
        "ecser": 0.004,
        "esser": 0.015,
        "ecdam": 0.018,
        "esdam": 0.060,
        "temperature": 40.0,
        "kLsp": 0.022,
        "iterMax": 1000,
        "ncl": 40,
        "tolerance": 0.001,
        "strainStep": 0.0001,
        "longSteelRatio": longSteelRatio,
        "transvSteelRatio": transvSteelRatio,
        "axialRatio": axialRatio,
    }


def runRectangularExample() -> AnalysisResults:
    """
    Run the rectangular example with default parameters and return results.
    """

    config = defaultRectangularConfig()
    return analyzeRectangularSection(config)


def runCircularExample() -> AnalysisResults:
    """
    Run the circular example with default parameters and return results.
    """

    config = defaultCircularConfig()
    return analyzeCircularSection(config)


if __name__ == "__main__":
    # Example execution for both geometries.
    rectResults = runRectangularExample()
    circResults = runCircularExample()
    plt.show()
