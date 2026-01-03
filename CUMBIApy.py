#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================================
                                    CUMBIApy v1.0

          SECTION AND MEMBER RESPONSE OF RC MEMBERS (Python Version)

                     Converted from MATLAB by AI Assistant
                     Original MATLAB code by Luis A. Montejo

              Updates available at https://github.com/LuisMontejo/CUMBIA

Cite as: Montejo, L. A., & Kowalsky, M. J. (2007). CUMBIA—Set of codes for the
         analysis of reinforced concrete members. CFL technical rep. no. IS-07, 1.

                            Last updated: 2026-01-02
==========================================================================================

This module provides classes and methods for analyzing reinforced concrete members
including:
    - Circular and rectangular cross-sections
    - Material models (Mander confined/unconfined concrete, King/Raynor steel)
    - Moment-curvature analysis
    - Force-displacement analysis
    - Interaction diagrams (P-M)
    - Buckling models (Moyer-Kowalsky, Berry-Eberhard)
    - Shear capacity assessment

Example:
    >>> from CUMBIApy import CircularSection
    >>> section = CircularSection(
    ...     diameter=1000,
    ...     cover=50,
    ...     numLongBars=22,
    ...     longBarDiam=25,
    ...     transBarDiam=9,
    ...     spacing=120
    ... )
    >>> section.setMaterialProperties(fpc=35, fy=460, fyh=400)
    >>> section.setMemberProperties(length=3000, bending='single')
    >>> section.analyze(axialLoad=400)
    >>> section.plotResults()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings

# =============================================================================
# MATERIAL MODELS
# =============================================================================

class MaterialModels:
    """
    Material constitutive models for concrete and steel.

    This class provides static methods for generating stress-strain curves
    for confined concrete, unconfined concrete, and reinforcing steel using
    various material models.

    Available models:
        - Mander model for confined concrete (normal and lightweight)
        - Mander model for unconfined concrete (normal and lightweight)
        - King model for steel
        - Raynor model for steel
    """

    @staticmethod
    def manderConfined(ec, ast, dh, clb, s, fpc, fy, eco, esm, espall,
                       section, d=0, height=0, width=0, ncx=0, ncy=0, wi=None,
                       dels=0.0001, transType='spirals'):
        """
        Mander model for confined concrete (normal weight).

        This function calculates the stress-strain curve for confined concrete
        based on the Mander et al. (1988) model.

        Parameters:
            ec : float
                Elastic modulus of concrete (MPa)
            ast : float
                Total area of longitudinal steel (mm²)
            dh : float
                Diameter of transverse reinforcement (mm)
            clb : float
                Cover to longitudinal bars (mm)
            s : float
                Spacing of transverse steel (mm)
            fpc : float
                Unconfined concrete compressive strength (MPa)
            fy : float
                Yield strength of transverse steel (MPa)
            eco : float
                Strain at peak unconfined concrete stress (typically 0.002)
            esm : float
                Maximum transverse steel strain (typically 0.10-0.15)
            espall : float
                Maximum unconfined concrete strain (typically 0.0064)
            section : str
                Section type ('circular' or 'rectangular')
            d : float, optional
                Diameter for circular section (mm)
            height : float, optional
                Height for rectangular section (mm)
            width : float, optional
                Width for rectangular section (mm)
            ncx : int, optional
                Number of legs of transverse steel in x-direction
            ncy : int, optional
                Number of legs of transverse steel in y-direction
            wi : array-like, optional
                Clear distances between longitudinal bars
            dels : float, optional
                Strain increment for curve generation (default: 0.0001)
            transType : str, optional
                Type of transverse reinforcement ('spirals' or 'hoops')

        Returns:
            strainConf : ndarray
                Array of strain values
            stressConf : ndarray
                Array of stress values (MPa)

        References:
            Mander, J. B., Priestley, M. J., & Park, R. (1988). Theoretical
            stress-strain model for confined concrete. Journal of structural
            engineering, 114(8), 1804-1826.
        """
        sp = s - dh
        ash = 0.25 * np.pi * (dh**2)

        if section.lower() == 'rectangular':
            bc = width - 2*clb + dh
            dc = height - 2*clb + dh
            asx = ncx * ash
            asy = ncy * ash
            ac = bc * dc
            rocc = ast / ac
            rox = asx / (s * dc)
            roy = asy / (s * bc)
            ros = rox + roy

            if wi is None:
                wi = np.array([])
            ke = ((1 - np.sum(wi**2) / (6*bc*dc)) * (1 - sp/(2*bc)) *
                  (1 - sp/(2*dc))) / (1 - rocc)
            ro = 0.5 * ros
            fpl = ke * ro * fy

        elif section.lower() == 'circular':
            ds = d - 2*clb + dh
            ros = 4 * ash / (ds * s)
            ac = 0.25 * np.pi * (ds**2)
            rocc = ast / ac

            if transType.lower() == 'spirals':
                ke = (1 - sp/(2*ds)) / (1 - rocc)
            elif transType.lower() == 'hoops':
                ke = ((1 - sp/(2*ds)) / (1 - rocc))**2
            else:
                raise ValueError("Transverse reinforcement should be 'spirals' or 'hoops'")

            fpl = 0.5 * ke * ros * fy
        else:
            raise ValueError("Section type not available. Use 'circular' or 'rectangular'")

        # Confined concrete properties
        fpcc = (-1.254 + 2.254*np.sqrt(1 + 7.94*fpl/fpc) - 2*fpl/fpc) * fpc
        ecc = eco * (1 + 5*(fpcc/fpc - 1))
        esec = fpcc / ecc
        r = ec / (ec - esec)
        ecu = 1.5 * (0.004 + 1.4*ros*fy*esm/fpcc)

        # Generate stress-strain curve
        strainConf = np.arange(0, ecu + dels, dels)
        x = strainConf / ecc
        stressConf = fpcc * x * r / (r - 1 + x**r)

        return strainConf, stressConf

    @staticmethod
    def manderConfinedLightweight(ec, ast, dh, clb, s, fpc, fy, eco, esm, espall,
                                   section, d=0, height=0, width=0, ncx=0, ncy=0,
                                   wi=None, dels=0.0001, transType='spirals'):
        """
        Mander model for confined lightweight concrete.

        Similar to manderConfined but uses a different equation for fpcc
        suitable for lightweight concrete.

        Parameters:
            Same as manderConfined()

        Returns:
            strainConf : ndarray
                Array of strain values
            stressConf : ndarray
                Array of stress values (MPa)
        """
        sp = s - dh
        ash = 0.25 * np.pi * (dh**2)

        if section.lower() == 'rectangular':
            bc = width - 2*clb + dh
            dc = height - 2*clb + dh
            asx = ncx * ash
            asy = ncy * ash
            ac = bc * dc
            rocc = ast / ac
            rox = asx / (s * dc)
            roy = asy / (s * bc)
            ros = rox + roy

            if wi is None:
                wi = np.array([])
            ke = ((1 - np.sum(wi**2) / (6*bc*dc)) * (1 - sp/(2*bc)) *
                  (1 - sp/(2*dc))) / (1 - rocc)
            ro = 0.5 * ros
            fpl = ke * ro * fy

        elif section.lower() == 'circular':
            ds = d - 2*clb + dh
            ros = 4 * ash / (ds * s)
            ac = 0.25 * np.pi * (ds**2)
            rocc = ast / ac

            if transType.lower() == 'spirals':
                ke = (1 - sp/(2*ds)) / (1 - rocc)
            elif transType.lower() == 'hoops':
                ke = ((1 - sp/(2*ds)) / (1 - rocc))**2
            else:
                raise ValueError("Transverse reinforcement should be 'spirals' or 'hoops'")

            fpl = 0.5 * ke * ros * fy
        else:
            raise ValueError("Section type not available")

        # Lightweight concrete model
        fpcc = (1 + fpl/(2*fpc)) * fpc
        ecc = eco * (1 + 5*(fpcc/fpc - 1))
        esec = fpcc / ecc
        r = ec / (ec - esec)
        ecu = 1.5 * (0.004 + 1.4*ros*fy*esm/fpcc)

        # Generate stress-strain curve
        strainConf = np.arange(0, ecu + dels, dels)
        x = strainConf / ecc
        stressConf = fpcc * x * r / (r - 1 + x**r)

        return strainConf, stressConf

    @staticmethod
    def manderUnconfined(ec, ast, dh, clb, s, fpc, fyh, eco, esm, espall,
                        section, d=0, height=0, width=0, ncx=0, ncy=0,
                        wi=None, dels=0.0001):
        """
        Mander model for unconfined concrete.

        This function generates the stress-strain curve for unconfined concrete
        including the softening branch after peak stress.

        Parameters:
            ec : float
                Elastic modulus of concrete (MPa)
            ast : float
                Total area of longitudinal steel (mm²)
            dh : float
                Diameter of transverse reinforcement (mm)
            clb : float
                Cover to longitudinal bars (mm)
            s : float
                Spacing of transverse steel (mm)
            fpc : float
                Unconfined concrete compressive strength (MPa)
            fyh : float
                Yield strength of transverse steel (MPa)
            eco : float
                Strain at peak stress (typically 0.002)
            esm : float
                Maximum transverse steel strain
            espall : float
                Spalling strain (typically 0.0064)
            section : str
                Section type ('circular' or 'rectangular')
            d, height, width, ncx, ncy, wi : optional
                Section geometry parameters (not used in unconfined model)
            dels : float, optional
                Strain increment (default: 0.0001)

        Returns:
            strainUnconf : ndarray
                Array of strain values
            stressUnconf : ndarray
                Array of stress values (MPa)
        """
        strainUnconf = np.arange(0, espall + dels, dels)
        esecUnconf = fpc / eco
        ru = ec / (ec - esecUnconf)
        xu = strainUnconf / eco

        stressUnconf = np.zeros_like(strainUnconf)

        for i in range(len(strainUnconf)):
            if strainUnconf[i] < 2*eco:
                stressUnconf[i] = fpc * xu[i] * ru / (ru - 1 + xu[i]**ru)
            elif strainUnconf[i] >= 2*eco and strainUnconf[i] <= espall:
                stressUnconf[i] = (fpc * (2*ru / (ru - 1 + 2**ru)) *
                                   (1 - (strainUnconf[i] - 2*eco) / (espall - 2*eco)))
            else:
                stressUnconf[i] = 0

        return strainUnconf, stressUnconf

    @staticmethod
    def manderUnconfinedLightweight(ec, nbl, dbl, dh, clb, s, fpc, fyh, eco,
                                     esm, espall, section, d=0, height=0, width=0,
                                     ncx=0, ncy=0, wi=None, dels=0.0001):
        """
        Mander model for unconfined lightweight concrete.

        Parameters:
            ec : float
                Elastic modulus of concrete (MPa)
            nbl : int
                Number of longitudinal bars
            dbl : float
                Diameter of longitudinal bars (mm)
            Other parameters same as manderUnconfined()

        Returns:
            strainUnconf : ndarray
                Array of strain values
            stressUnconf : ndarray
                Array of stress values (MPa)
        """
        strainUnconf = np.arange(0, espall + dels, dels)
        esecUnconf = fpc / eco
        ru = ec / (ec - esecUnconf)
        xu = strainUnconf / eco
        ru2 = ec / (ec - 1.8*fpc/eco)

        stressUnconf = np.zeros_like(strainUnconf)

        for i in range(len(strainUnconf)):
            if strainUnconf[i] < eco:
                stressUnconf[i] = fpc * xu[i] * ru / (ru - 1 + xu[i]**ru)
            elif strainUnconf[i] >= eco and strainUnconf[i] < 1.3*eco:
                stressUnconf[i] = fpc * xu[i] * ru2 / (ru2 - 1 + xu[i]**ru2)
            elif strainUnconf[i] >= 1.3*eco and strainUnconf[i] <= espall:
                stressUnconf[i] = (fpc * (1.3*ru2 / (ru2 - 1 + 1.3**ru2)) *
                                   (1 - (strainUnconf[i] - 1.3*eco) / (espall - 1.3*eco)))
            else:
                stressUnconf[i] = 0

        return strainUnconf, stressUnconf

    @staticmethod
    def steelKing(es, fy, fsu, esh, esu, dels=0.0001):
        """
        King et al. model for reinforcing steel.

        This model includes elastic range, yield plateau, and strain hardening.

        Parameters:
            es : float
                Elastic modulus of steel (MPa)
            fy : float
                Yield stress (MPa)
            fsu : float
                Ultimate stress (MPa)
            esh : float
                Strain at start of strain hardening (typically 0.008)
            esu : float
                Ultimate strain (typically 0.10-0.15)
            dels : float, optional
                Strain increment (default: 0.0001)

        Returns:
            strainSteel : ndarray
                Array of strain values
            stressSteel : ndarray
                Array of stress values (MPa)

        References:
            King, D. J., Priestley, M. J. N., & Park, R. (1986). Computer
            programs for concrete column design. Department of Civil Engineering,
            University of Canterbury.
        """
        r = esu - esh
        m = ((fsu/fy)*((30*r + 1)**2) - 60*r - 1) / (15 * (r**2))
        strainSteel = np.arange(0, esu + dels, dels)
        ey = fy / es

        stressSteel = np.zeros_like(strainSteel)

        for i in range(len(strainSteel)):
            if strainSteel[i] < ey:
                stressSteel[i] = es * strainSteel[i]
            elif strainSteel[i] >= ey and strainSteel[i] <= esh:
                stressSteel[i] = fy
            else:  # strainSteel[i] > esh
                stressSteel[i] = ((m*(strainSteel[i] - esh) + 2) /
                                  (60*(strainSteel[i] - esh) + 2) +
                                  (strainSteel[i] - esh)*(60 - m) /
                                  (2*((30*r + 1)**2))) * fy

        return strainSteel, stressSteel

    @staticmethod
    def steelRaynor(es, fy, fsu, esh, esu, dels=0.0001, c1=3.5, ey_plateau=350):
        """
        Raynor et al. model for reinforcing steel.

        This model provides a more refined representation of the yield plateau
        and strain hardening behavior.

        Parameters:
            es : float
                Elastic modulus of steel (MPa)
            fy : float
                Yield stress (MPa)
            fsu : float
                Ultimate stress (MPa)
            esh : float
                Strain at start of strain hardening (typically 0.008)
            esu : float
                Ultimate strain (typically 0.10-0.15)
            dels : float, optional
                Strain increment (default: 0.0001)
            c1 : float, optional
                Parameter defining strain hardening curve shape (typically 2-6)
                Default: 3.5
            ey_plateau : float, optional
                Slope of yield plateau (MPa)
                Default: 350

        Returns:
            strainSteel : ndarray
                Array of strain values
            stressSteel : ndarray
                Array of stress values (MPa)

        References:
            Raynor, D. J., Lehman, D. E., & Stanton, J. F. (2002). Bond-slip
            response of reinforcing bars grouted in ducts. ACI Structural
            Journal, 99(5), 568-576.
        """
        strainSteel = np.arange(0, esu + dels, dels)
        ey = fy / es
        fsh = fy + (esh - ey) * ey_plateau

        stressSteel = np.zeros_like(strainSteel)

        for i in range(len(strainSteel)):
            if strainSteel[i] < ey:
                # Elastic range
                stressSteel[i] = es * strainSteel[i]
            elif strainSteel[i] >= ey and strainSteel[i] <= esh:
                # Yield plateau with slight hardening
                stressSteel[i] = fy + (strainSteel[i] - ey) * ey_plateau
            else:  # strainSteel[i] > esh
                # Strain hardening
                stressSteel[i] = fsu - (fsu - fsh) * (((esu - strainSteel[i]) /
                                                        (esu - esh))**c1)

        return strainSteel, stressSteel


# =============================================================================
# BASE SECTION CLASS
# =============================================================================

class RCSection:
    """
    Base class for reinforced concrete sections.

    This abstract base class provides common functionality for both circular
    and rectangular RC sections including material model generation, analysis
    setup, and result storage.

    Attributes:
        results : dict
            Dictionary containing analysis results
        materialProps : dict
            Material property values
        geometryProps : dict
            Section geometry parameters
    """

    def __init__(self):
        """Initialize the RC section with empty containers for properties and results."""
        self.results = {}
        self.materialProps = {}
        self.geometryProps = {}
        self.memberProps = {}
        self.analysisParams = {
            'iterMax': 1000,
            'numConcreteLayers': 40,
            'tolerance': 0.001,
            'deltaStrain': 0.0001
        }

    def _interpolateStress(self, strain, strainArray, stressArray):
        """
        Interpolate stress value at given strain.

        Parameters:
            strain : float or ndarray
                Strain value(s) at which to interpolate
            strainArray : ndarray
                Array of strain values from material model
            stressArray : ndarray
                Array of stress values from material model

        Returns:
            float or ndarray
                Interpolated stress value(s)
        """
        return np.interp(strain, strainArray, stressArray)

    def _createDeformationVector(self, ecu):
        """
        Create vector of deformation values for analysis.

        The deformation vector has variable spacing to efficiently capture
        the response at critical points.

        Parameters:
            ecu : float
                Ultimate concrete strain

        Returns:
            ndarray
                Array of deformation values
        """
        if ecu <= 0.0018:
            deformation = np.arange(0.0001, 20*ecu, 0.0001)
        elif ecu > 0.0018 and ecu <= 0.0025:
            deformation = np.concatenate([
                np.arange(0.0001, 0.0016, 0.0001),
                np.arange(0.0018, 20*ecu, 0.0002)
            ])
        elif ecu > 0.0025 and ecu <= 0.006:
            deformation = np.concatenate([
                np.arange(0.0001, 0.0016, 0.0001),
                np.arange(0.0018, 0.002, 0.0002),
                np.arange(0.0025, 20*ecu, 0.0005)
            ])
        elif ecu > 0.006 and ecu <= 0.012:
            deformation = np.concatenate([
                np.arange(0.0001, 0.0016, 0.0001),
                np.arange(0.0018, 0.002, 0.0002),
                np.arange(0.0025, 0.005, 0.0005),
                np.arange(0.006, 20*ecu, 0.001)
            ])
        else:  # ecu > 0.012
            deformation = np.concatenate([
                np.arange(0.0001, 0.0016, 0.0001),
                np.arange(0.0018, 0.002, 0.0002),
                np.arange(0.0025, 0.005, 0.0005),
                np.arange(0.006, 0.01, 0.001),
                np.arange(0.012, 20*ecu, 0.002)
            ])

        return deformation

    def plotMaterialModels(self):
        """
        Plot the stress-strain curves for all materials.

        Creates two figures:
            1. Confined and unconfined concrete
            2. Reinforcing steel
        """
        if not hasattr(self, 'strainConf'):
            raise ValueError("Material models not generated. Call analyze() first.")

        # Plot concrete
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.fill_between(self.strainConf, self.stressConf, alpha=0.5,
                         color='cyan', label='Confined Concrete')
        ax1.fill_between(self.strainUnconf, self.stressUnconf, alpha=0.5,
                         color='blue', label='Unconfined Concrete')
        ax1.set_xlabel('Strain', fontsize=14)
        ax1.set_ylabel('Stress [MPa]', fontsize=14)
        ax1.set_title('Stress-Strain Relation for Confined and Unconfined Concrete',
                     fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([self.strainConf[0], self.strainConf[-1]])

        # Plot steel
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.fill_between(self.strainSteel, self.stressSteel, alpha=0.5,
                         color='gold', label='Reinforcing Steel')
        ax2.set_xlabel('Strain', fontsize=14)
        ax2.set_ylabel('Stress [MPa]', fontsize=14)
        ax2.set_title('Stress-Strain Relation for Reinforcing Steel', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig1, fig2


# =============================================================================
# CIRCULAR SECTION CLASS
# =============================================================================

class CircularSection(RCSection):
    """
    Circular reinforced concrete section analysis.

    This class performs comprehensive analysis of circular RC members including:
        - Moment-curvature analysis
        - Force-displacement analysis
        - P-M interaction diagrams
        - Buckling assessment (Moyer-Kowalsky and Berry-Eberhard models)
        - Shear capacity evaluation
        - Deformation limit states

    Example:
        >>> section = CircularSection(
        ...     diameter=1000,
        ...     cover=50,
        ...     numLongBars=22,
        ...     longBarDiam=25,
        ...     transBarDiam=9,
        ...     spacing=120,
        ...     transType='spirals'
        ... )
        >>> section.setMaterialProperties(
        ...     fpc=35,
        ...     fy=460,
        ...     fyh=400,
        ...     fsu=620,
        ...     concreteModel='mc',
        ...     steelModel='ra'
        ... )
        >>> section.setMemberProperties(length=3000, bending='single')
        >>> section.analyze(axialLoad=400)
        >>> section.writeResults('output.txt')
        >>> section.plotResults()
    """

    def __init__(self, diameter, cover, numLongBars, longBarDiam,
                 transBarDiam, spacing, transType='spirals'):
        """
        Initialize circular section geometry.

        Parameters:
            diameter : float
                Section diameter (mm)
            cover : float
                Cover to longitudinal bars (mm)
            numLongBars : int
                Number of longitudinal bars
            longBarDiam : float
                Longitudinal bar diameter (mm)
            transBarDiam : float
                Transverse reinforcement diameter (mm)
            spacing : float
                Spacing of transverse steel (mm)
            transType : str, optional
                Type of transverse reinforcement ('spirals' or 'hoops')
                Default: 'spirals'
        """
        super().__init__()

        # Store geometry
        self.geometryProps = {
            'diameter': diameter,
            'cover': cover,
            'numLongBars': numLongBars,
            'longBarDiam': longBarDiam,
            'transBarDiam': transBarDiam,
            'spacing': spacing,
            'transType': transType
        }

        # Calculate derived geometry
        self.dsp = diameter - 2*cover + transBarDiam  # Core diameter
        self.dcore = cover - transBarDiam*0.5  # Distance to core
        self.ast = numLongBars * 0.25 * np.pi * (longBarDiam**2)  # Total steel area
        self.agross = 0.25 * np.pi * (diameter**2)  # Gross area

    def setMaterialProperties(self, fpc, fy, fyh, es=200000, fsu=None,
                             esh=0.008, esu=0.12, eco=0.002, esm=0.11,
                             espall=0.0064, ec=None, c1=3.5, eyPlateau=350,
                             concreteModel='mc', steelModel='ra',
                             lightweightConcrete=False):
        """
        Set material properties and generate constitutive models.

        Parameters:
            fpc : float
                Concrete compressive strength (MPa)
            fy : float
                Longitudinal steel yield stress (MPa)
            fyh : float
                Transverse steel yield stress (MPa)
            es : float, optional
                Steel elastic modulus (MPa). Default: 200000
            fsu : float, optional
                Ultimate steel stress (MPa). If None, defaults to 1.35*fy
            esh : float, optional
                Strain at strain hardening (typically 0.008). Default: 0.008
            esu : float, optional
                Ultimate steel strain (typically 0.10-0.15). Default: 0.12
            eco : float, optional
                Unconfined concrete peak strain. Default: 0.002
            esm : float, optional
                Maximum transverse steel strain. Default: 0.11
            espall : float, optional
                Spalling strain. Default: 0.0064
            ec : float, optional
                Concrete elastic modulus (MPa). If None, calculated as 5000*sqrt(fpc)
            c1 : float, optional
                Raynor model hardening parameter (2-6). Default: 3.5
            eyPlateau : float, optional
                Raynor model yield plateau slope (MPa). Default: 350
            concreteModel : str, optional
                'mc' for Mander confined, 'mu' for Mander unconfined,
                'mclw' for Mander confined lightweight. Default: 'mc'
            steelModel : str, optional
                'ra' for Raynor, 'ks' for King. Default: 'ra'
            lightweightConcrete : bool, optional
                Use lightweight concrete models. Default: False
        """
        # Set defaults
        if fsu is None:
            fsu = 1.35 * fy
        if ec is None:
            ec = 5000 * np.sqrt(fpc)

        # Store material properties
        self.materialProps = {
            'fpc': fpc,
            'fy': fy,
            'fyh': fyh,
            'es': es,
            'fsu': fsu,
            'esh': esh,
            'esu': esu,
            'eco': eco,
            'esm': esm,
            'espall': espall,
            'ec': ec,
            'c1': c1,
            'eyPlateau': eyPlateau,
            'concreteModel': concreteModel,
            'steelModel': steelModel,
            'lightweightConcrete': lightweightConcrete
        }

        # Generate material models
        self._generateMaterialModels()

    def _generateMaterialModels(self):
        """Generate stress-strain curves for all materials."""
        geom = self.geometryProps
        mat = self.materialProps

        # Confined concrete
        if mat['concreteModel'].lower() == 'mc':
            if mat['lightweightConcrete']:
                self.strainConf, self.stressConf = MaterialModels.manderConfinedLightweight(
                    mat['ec'], self.ast, geom['transBarDiam'], geom['cover'],
                    geom['spacing'], mat['fpc'], mat['fyh'], mat['eco'], mat['esm'],
                    mat['espall'], 'circular', d=geom['diameter'],
                    dels=self.analysisParams['deltaStrain'],
                    transType=geom['transType']
                )
            else:
                self.strainConf, self.stressConf = MaterialModels.manderConfined(
                    mat['ec'], self.ast, geom['transBarDiam'], geom['cover'],
                    geom['spacing'], mat['fpc'], mat['fyh'], mat['eco'], mat['esm'],
                    mat['espall'], 'circular', d=geom['diameter'],
                    dels=self.analysisParams['deltaStrain'],
                    transType=geom['transType']
                )
        else:
            # User can provide custom model
            raise NotImplementedError("Custom material models not yet implemented")

        # Unconfined concrete
        if mat['lightweightConcrete']:
            self.strainUnconf, self.stressUnconf = MaterialModels.manderUnconfinedLightweight(
                mat['ec'], geom['numLongBars'], geom['longBarDiam'],
                geom['transBarDiam'], geom['cover'], geom['spacing'],
                mat['fpc'], mat['fyh'], mat['eco'], mat['esm'], mat['espall'],
                'circular', d=geom['diameter'], dels=self.analysisParams['deltaStrain']
            )
        else:
            self.strainUnconf, self.stressUnconf = MaterialModels.manderUnconfined(
                mat['ec'], self.ast, geom['transBarDiam'], geom['cover'],
                geom['spacing'], mat['fpc'], mat['fyh'], mat['eco'], mat['esm'],
                mat['espall'], 'circular', d=geom['diameter'],
                dels=self.analysisParams['deltaStrain']
            )

        # Steel model
        if mat['steelModel'].lower() == 'ra':
            self.strainSteel, self.stressSteel = MaterialModels.steelRaynor(
                mat['es'], mat['fy'], mat['fsu'], mat['esh'], mat['esu'],
                dels=self.analysisParams['deltaStrain'], c1=mat['c1'],
                ey_plateau=mat['eyPlateau']
            )
        elif mat['steelModel'].lower() == 'ks':
            self.strainSteel, self.stressSteel = MaterialModels.steelKing(
                mat['es'], mat['fy'], mat['fsu'], mat['esh'], mat['esu'],
                dels=self.analysisParams['deltaStrain']
            )
        else:
            raise ValueError("Steel model must be 'ra' (Raynor) or 'ks' (King)")

        # Extend stress-strain curves for interpolation
        self.ecu = self.strainConf[-1]
        self.ecuMander = self.ecu / 1.5

        # Extend concrete arrays
        dels = self.analysisParams['deltaStrain']
        self.strainConf = np.concatenate([[-1e10], self.strainConf,
                                          [self.strainConf[-1] + dels, 1e10]])
        self.stressConf = np.concatenate([[0], self.stressConf, [0, 0]])

        self.strainUnconf = np.concatenate([[-1e10], self.strainUnconf,
                                            [self.strainUnconf[-1] + dels, 1e10]])
        self.stressUnconf = np.concatenate([[0], self.stressUnconf, [0, 0]])

        # Extend steel arrays (make symmetric for tension/compression)
        self.esu = self.strainSteel[-1]
        self.strainSteel = np.concatenate([self.strainSteel,
                                           [self.strainSteel[-1] + dels, 1e10]])
        self.stressSteel = np.concatenate([self.stressSteel, [0, 0]])

        # Create negative side
        strainSteelNeg = -self.strainSteel[::-1]
        stressSteelNeg = -self.stressSteel[::-1]
        self.strainSteel = np.concatenate([strainSteelNeg, self.strainSteel[1:]])
        self.stressSteel = np.concatenate([stressSteelNeg, self.stressSteel[1:]])

    def setMemberProperties(self, length, bending='single', ductilityMode='biaxial',
                           temperature=40, kLsp=0.022):
        """
        Set member-level properties.

        Parameters:
            length : float
                Member clear length (mm)
            bending : str, optional
                'single' or 'double' curvature. Default: 'single'
            ductilityMode : str, optional
                'biaxial' or 'uniaxial'. Default: 'biaxial'
            temperature : float, optional
                Temperature in Celsius. Default: 40
            kLsp : float, optional
                Constant for strain penetration length (Lsp = kLsp*fy*Dbl)
                Typically 0.022 at ambient temp, 0.011 at -40C. Default: 0.022
        """
        self.memberProps = {
            'length': length,
            'bending': bending.lower(),
            'ductilityMode': ductilityMode.lower(),
            'temperature': temperature,
            'kLsp': kLsp
        }

    def setLimitStates(self, concreteServiceStrain=0.004, steelServiceStrain=0.015,
                      concreteDamageStrain=0.018, steelDamageStrain=0.060,
                      concreteInteractionStrain=0.004, steelInteractionStrain=0.015):
        """
        Set deformation limit state criteria.

        Parameters:
            concreteServiceStrain : float, optional
                Concrete serviceability strain. Default: 0.004
            steelServiceStrain : float, optional
                Steel serviceability strain. Default: 0.015
            concreteDamageStrain : float or str, optional
                Concrete damage control strain. Use 'twth' for 2/3 of ultimate.
                Default: 0.018
            steelDamageStrain : float, optional
                Steel damage control strain. Default: 0.060
            concreteInteractionStrain : float, optional
                Concrete strain for interaction diagram. Default: 0.004
            steelInteractionStrain : float, optional
                Steel strain for interaction diagram. Default: 0.015
        """
        # Handle 'twth' option for damage strain
        if isinstance(concreteDamageStrain, str) and concreteDamageStrain.lower() == 'twth':
            concreteDamageStrain = self.ecuMander

        self.limitStates = {
            'ecser': concreteServiceStrain,
            'esser': -steelServiceStrain,
            'ecdam': concreteDamageStrain,
            'esdam': -steelDamageStrain,
            'csid': concreteInteractionStrain,
            'ssid': steelInteractionStrain
        }


    def _setupConcreteLayers(self):
        """
        Setup concrete layers for fiber section analysis.

        Creates layers of confined and unconfined concrete based on the
        section geometry. Each layer has an area and centroidal location.

        Sets:
            self.concreteLayers : ndarray
                Array with columns [centroid, area_unconf, area_conf, top_distance, rebar_area]
        """
        geom = self.geometryProps
        diameter = geom['diameter']
        ncl = self.analysisParams['numConcreteLayers']

        # Initial layer boundaries
        tcl = diameter / ncl
        yl = np.arange(tcl, diameter + tcl, tcl)

        # Add boundaries for core
        yl = np.sort(np.concatenate([yl, [self.dcore, diameter - self.dcore]]))

        # Remove duplicates
        yl = np.unique(yl)

        # Confined concrete layer boundaries
        yc = yl - self.dcore
        yc = yc[(yc > 0) & (yc < self.dsp)]
        yc = np.append(yc, self.dsp)

        # Calculate layer areas
        r = diameter / 2
        areaAux = ((r**2) * np.arccos(1 - 2*yl/diameter) -
                   (r - yl) * np.sqrt(diameter*yl - yl**2))
        atc = np.diff(np.concatenate([[0], areaAux]))

        # Confined layer areas
        rc = self.dsp / 2
        areaAux = ((rc**2) * np.arccos(1 - 2*yc/self.dsp) -
                   (rc - yc) * np.sqrt(self.dsp*yc - yc**2))
        atcc = np.diff(np.concatenate([[0], areaAux]))

        # Assign areas to confined/unconfined
        conclay = np.zeros((len(yl), 2))
        k = 0
        for i in range(len(yl)):
            if yl[i] <= self.dcore or yl[i] > diameter - self.dcore:
                conclay[i, :] = [atc[i], 0]  # Unconfined only
            else:
                conclay[i, :] = [atc[i] - atcc[k], atcc[k]]
                k += 1

        # Calculate centroids
        centroids = np.concatenate([[yl[0]/2],
                                    0.5*(yl[:-1] + yl[1:])])

        # Combine: [centroid, area_unconf, area_conf, top_distance, rebar_area]
        self.concreteLayers = np.column_stack([
            centroids,
            conclay[:, 0],
            conclay[:, 1],
            yl,
            np.zeros(len(yl))  # Rebar area (filled later)
        ])

    def _setupRebars(self):
        """
        Setup longitudinal reinforcement positions and areas.

        Calculates the position of each longitudinal bar and assigns
        bars to concrete layers.

        Sets:
            self.rebarDist : ndarray
                Distance from top of section to each bar
            self.rebarAreas : ndarray
                Area of each bar
        """
        geom = self.geometryProps
        diameter = geom['diameter']
        cover = geom['cover']
        nbl = geom['numLongBars']
        dbl = geom['longBarDiam']

        # Bar area
        asb = 0.25 * np.pi * (dbl**2)

        # Radius to bar centroids
        r = 0.5 * (diameter - 2*cover - dbl)

        # Angular positions
        theta = (2*np.pi / nbl) * np.arange(nbl)

        # Y-coordinates (distance from top)
        self.rebarDist = np.sort(0.5*(diameter - 2*r) + r*np.sin(theta)*np.tan(0.5*theta))
        self.rebarAreas = asb * np.ones(nbl)

        # Assign rebars to concrete layers and correct areas
        for k in range(1, len(self.concreteLayers) - 1):
            rebarInLayer = ((self.rebarDist <= self.concreteLayers[k, 3]) &
                           (self.rebarDist > self.concreteLayers[k-1, 3]))
            self.concreteLayers[k, 4] = np.sum(self.rebarAreas[rebarInLayer])

            # Correct concrete area for rebar presence
            if self.concreteLayers[k, 2] == 0:  # Unconfined layer
                self.concreteLayers[k, 1] -= self.concreteLayers[k, 4]
                if self.concreteLayers[k, 1] < 0:
                    raise ValueError("Negative concrete area. Decrease number of layers.")
            else:  # Confined layer
                self.concreteLayers[k, 2] -= self.concreteLayers[k, 4]
                if self.concreteLayers[k, 2] < 0:
                    raise ValueError("Negative concrete area. Decrease number of layers.")

    def _computeMomentCurvature(self, axialLoad):
        """
        Compute moment-curvature relationship using iterative section analysis.

        This is the core analysis routine that finds the neutral axis depth
        for each top fiber strain, ensuring force equilibrium.

        Parameters:
            axialLoad : float
                Applied axial force (kN). Positive = compression, Negative = tension

        Returns:
            dict : Dictionary containing:
                - curvature: Curvature values (1/m)
                - moment: Moment values (kN-m)
                - neutralAxis: Neutral axis depth (mm)
                - coverStrain: Top concrete strain
                - coreStrain: Core concrete strain
                - steelStrain: Extreme steel strain
                - message: Termination message
        """
        geom = self.geometryProps
        mat = self.materialProps
        diameter = geom['diameter']

        # Convert axial load to Newtons
        P = axialLoad * 1000

        # Get deformation vector
        deformation = self._createDeformationVector(self.ecu)

        # Check if initial deformation provides enough compression
        if P > 0:
            newDef = []
            for defVal in deformation:
                stressUnconf = self._interpolateStress(
                    defVal * np.ones(len(self.concreteLayers)),
                    self.strainUnconf, self.stressUnconf
                )
                stressConf = self._interpolateStress(
                    defVal * np.ones(len(self.concreteLayers)),
                    self.strainConf, self.stressConf
                )
                stressSteel = self._interpolateStress(
                    defVal * np.ones(len(self.rebarDist)),
                    self.strainSteel, self.stressSteel
                )

                compCheck = (np.sum(stressUnconf * self.concreteLayers[:, 1]) +
                            np.sum(stressConf * self.concreteLayers[:, 2]) +
                            np.sum(self.rebarAreas[0] * stressSteel))

                if compCheck >= P:
                    newDef.append(defVal)

            deformation = np.array(newDef)

        np_points = len(deformation)

        # Initialize result arrays
        curvature = np.zeros(np_points + 1)
        moment = np.zeros(np_points + 1)
        neutralAxis = np.zeros(np_points + 1)
        forceError = np.zeros(np_points + 1)
        numIter = np.zeros(np_points + 1, dtype=int)
        coverStrain = np.zeros(np_points + 1)
        coreStrain = np.zeros(np_points + 1)
        steelStrain = np.zeros(np_points + 1)

        # Tolerance
        tol = self.analysisParams['tolerance'] * 0.25 * np.pi * (diameter**2) * mat['fpc']
        iterMax = self.analysisParams['iterMax']

        # Initial neutral axis guess
        x = diameter / 2
        message = 0  # 0=ok, 1=concrete, 2=steel, 3=iterations, 4=strength loss

        for k in range(np_points):
            # Check for strength loss
            lostMomControl = np.max(moment[:k+1])
            if k > 0 and moment[k] < 0.8 * lostMomControl:
                message = 4
                break

            # Iterative search for neutral axis
            F = 10 * tol
            niter = 0

            while abs(F) > tol:
                niter += 1

                # Strains in concrete layers
                strainConcrete = (deformation[k] / x) * (
                    self.concreteLayers[:, 0] - (diameter - x)
                )

                # Strains in steel
                strainRebar = (deformation[k] / x) * (
                    self.rebarDist - (diameter - x)
                )

                # Stresses
                stressUnconf = self._interpolateStress(strainConcrete,
                                                       self.strainUnconf,
                                                       self.stressUnconf)
                stressConf = self._interpolateStress(strainConcrete,
                                                     self.strainConf,
                                                     self.stressConf)
                stressRebar = self._interpolateStress(strainRebar,
                                                      self.strainSteel,
                                                      self.stressSteel)

                # Forces
                forceUnconf = stressUnconf * self.concreteLayers[:, 1]
                forceConf = stressConf * self.concreteLayers[:, 2]
                forceSteel = self.rebarAreas[0] * stressRebar

                # Total force
                F = np.sum(forceUnconf) + np.sum(forceConf) + np.sum(forceSteel) - P

                # Update neutral axis
                if F > 0:
                    x = x - 0.05 * x
                else:
                    x = x + 0.05 * x

                if niter > iterMax:
                    message = 3
                    break

            if message == 3:
                break

            # Check strain limits
            cores = (deformation[k] / x) * abs(x - self.dcore)

            # Check if confined and unconfined models are same
            sameModel = (mat['concreteModel'] == 'mu')

            if not sameModel:
                if cores >= self.ecu:
                    message = 1
                    break
            else:
                if deformation[k] >= self.ecu:
                    message = 1
                    break

            if abs(strainRebar[0]) > self.esu:
                message = 2
                break

            # Store results
            neutralAxis[k+1] = x
            forceError[k+1] = F
            numIter[k+1] = niter

            # Moment about center
            momentVal = (np.sum(forceUnconf * self.concreteLayers[:, 0]) +
                        np.sum(forceConf * self.concreteLayers[:, 0]) +
                        np.sum(forceSteel * self.rebarDist) -
                        P * (diameter / 2)) / 1e6

            if momentVal < 0:
                momentVal = -0.01 * momentVal

            moment[k+1] = momentVal
            curvature[k+1] = 1000 * deformation[k] / x
            coverStrain[k+1] = deformation[k]
            coreStrain[k+1] = cores
            steelStrain[k+1] = strainRebar[0]

            if message != 0:
                break

        # Trim arrays to actual length
        lastIdx = k + 2 if message != 0 else np_points + 1

        return {
            'curvature': curvature[:lastIdx],
            'moment': moment[:lastIdx],
            'neutralAxis': neutralAxis[:lastIdx],
            'coverStrain': coverStrain[:lastIdx],
            'coreStrain': coreStrain[:lastIdx],
            'steelStrain': steelStrain[:lastIdx],
            'forceError': forceError[:lastIdx],
            'numIterations': numIter[:lastIdx],
            'message': message
        }


    def analyze(self, axialLoad=0, performInteraction=False):
        """
        Perform complete section and member analysis.

        This is the main analysis method that orchestrates all calculations including:
        - Moment-curvature analysis
        - Nominal moment capacity
        - Plastic hinge length
        - Force-displacement relationship
        - Shear capacity
        - Buckling assessment
        - Deformation limit states
        - P-M interaction diagram (if requested)

        Parameters:
            axialLoad : float, optional
                Applied axial force (kN). Positive = compression. Default: 0
            performInteraction : bool, optional
                Whether to compute P-M interaction diagram. Default: False

        Returns:
            dict : Complete analysis results
        """
        if not hasattr(self, 'memberProps'):
            raise ValueError("Member properties not set. Call setMemberProperties() first.")
        if not hasattr(self, 'limitStates'):
            self.setLimitStates()  # Use defaults

        print("Setting up fiber section...")
        self._setupConcreteLayers()
        self._setupRebars()

        print(f"Computing moment-curvature for P = {axialLoad} kN...")
        mcResults = self._computeMomentCurvature(axialLoad)

        # Store basic M-φ results
        self.results['momentCurvature'] = mcResults

        # Calculate section properties
        print("Computing section properties...")
        self._computeSectionProperties(mcResults, axialLoad)

        # Calculate member response
        print("Computing member force-displacement...")
        self._computeForceDisplacement(mcResults, axialLoad)

        # Buckling models
        print("Evaluating buckling models...")
        self._computeBucklingModels()

        # Deformation limit states
        print("Evaluating limit states...")
        self._computeLimitStates()

        # P-M interaction if requested
        if performInteraction:
            print("Computing P-M interaction diagram...")
            self._computeInteractionDiagram()

        print("Analysis complete!")
        return self.results

    def _computeSectionProperties(self, mcResults, axialLoad):
        """
        Compute section-level properties from M-φ results.

        Calculates:
        - Nominal moment and equivalent curvature
        - First yield point
        - Section curvature ductility
        - Bilinear approximation
        """
        geom = self.geometryProps
        mat = self.materialProps
        lim = self.limitStates

        curv = mcResults['curvature']
        mom = mcResults['moment']
        coverStrain = mcResults['coverStrain']
        steelStrain = mcResults['steelStrain']

        # Calculate tensile strength for cracking
        temp = self.memberProps['temperature']
        if temp < 0:
            ct = (1 - 0.0105*temp) * 0.56 * np.sqrt(mat['fpc'])
        else:
            ct = 0.56 * np.sqrt(mat['fpc'])
        eccr = ct / mat['ec']

        # Nominal moment (serviceability criteria)
        mn = np.interp(lim['ecser'], coverStrain, mom)
        esaux = np.interp(lim['ecser'], coverStrain, steelStrain)

        # Check if steel controls
        if abs(esaux) > abs(lim['esser']) or np.isnan(mn):
            mnSteel = np.interp(lim['esser'], steelStrain, mom)
            if not np.isnan(mnSteel):
                mn = mnSteel
            elif np.isnan(mn) and np.isnan(mnSteel):
                raise ValueError("Problem estimating nominal moment from serviceability strains")

        # Neutral axis depth at nominal moment
        cMn = np.interp(mn, mom, mcResults['neutralAxis'])

        # First yield curvature
        fycurvC = np.interp(1.8*mat['fpc']/mat['ec'], coverStrain, curv)
        fycurvS = np.interp(-mat['fy']/mat['es'], steelStrain, curv)
        fycurv = min(fycurvC, fycurvS)
        fyM = np.interp(fycurv, curv, mom)

        # Equivalent curvature
        eqcurv = max((mn/fyM)*fycurv, fycurv)

        # Bilinear approximation
        curvBilin = np.array([0, eqcurv, curv[-1]])
        momBilin = np.array([0, mn, mom[-1]])

        # Section curvature ductility
        sectionCurvatureDuctility = curv[-1] / eqcurv

        # Store results
        self.results['nominalMoment'] = mn
        self.results['equivalentCurvature'] = eqcurv
        self.results['yieldMoment'] = fyM
        self.results['yieldCurvature'] = fycurv
        self.results['neutralAxisAtMn'] = cMn
        self.results['sectionCurvatureDuctility'] = sectionCurvatureDuctility
        self.results['curvatureBilinear'] = curvBilin
        self.results['momentBilinear'] = momBilin
        self.results['crackingStrain'] = eccr

    def _computeForceDisplacement(self, mcResults, axialLoad):
        """
        Compute force-displacement relationship including shear and flexural deformations.

        This method calculates:
        - Plastic hinge length
        - Strain penetration length
        - Flexural displacement
        - Shear displacement
        - Total displacement
        - Displacement ductility
        - Shear capacity vs displacement
        """
        geom = self.geometryProps
        mat = self.materialProps
        mem = self.memberProps
        diameter = geom['diameter']
        L = mem['length']

        curv = mcResults['curvature']
        mom = mcResults['moment']
        coverStrain = mcResults['coverStrain']
        steelStrain = mcResults['steelStrain']

        # Get section properties
        mn = self.results['nominalMoment']
        fycurv = self.results['yieldCurvature']
        fyM = self.results['yieldMoment']
        eqcurv = self.results['equivalentCurvature']
        cMn = self.results['neutralAxisAtMn']
        eccr = self.results['crackingStrain']

        # Strain penetration length
        lsp = np.zeros(len(steelStrain))
        for j in range(len(steelStrain)):
            ffss = -steelStrain[j] * mat['es']
            if ffss > mat['fy']:
                ffss = mat['fy']
            lsp[j] = mem['kLsp'] * ffss * geom['longBarDiam']

        # Plastic hinge length
        kkk = min(0.2*(mat['fsu']/mat['fy'] - 1), 0.08)
        if mem['bending'] == 'single':
            lp = max(kkk*L + mem['kLsp']*mat['fy']*geom['longBarDiam'],
                    2*mem['kLsp']*mat['fy']*geom['longBarDiam'])
            lbe = L
        else:  # double
            lp = max(kkk*L/2 + mem['kLsp']*mat['fy']*geom['longBarDiam'],
                    2*mem['kLsp']*mat['fy']*geom['longBarDiam'])
            lbe = L / 2

        # Flexural displacement
        displF = np.zeros(len(curv))
        for i in range(len(curv)):
            if coverStrain[i] < eccr:
                # Uncracked
                if mem['bending'] == 'single':
                    displF[i] = curv[i] * ((L/1000)**2) / 3
                else:
                    displF[i] = curv[i] * ((L/1000)**2) / 6
            elif coverStrain[i] > eccr and curv[i] < fycurv:
                # Cracked elastic
                if mem['bending'] == 'single':
                    displF[i] = curv[i] * (((L + lsp[i])/1000)**2) / 3
                else:
                    displF[i] = curv[i] * (((L + 2*lsp[i])/1000)**2) / 6
            else:  # curv[i] >= fycurv
                # Post-yield
                if mem['bending'] == 'single':
                    displF[i] = ((curv[i] - fycurv*(mom[i]/fyM)) * (lp/1000) *
                                ((L + lsp[i] - 0.5*lp)/1000) +
                                (fycurv * (((L + lsp[i])/1000)**2) / 3) * (mom[i]/fyM))
                else:
                    displF[i] = ((curv[i] - fycurv*(mom[i]/fyM)) * (lp/1000) *
                                ((L + 2*(lsp[i] - 0.5*lp))/1000) +
                                (fycurv * (((L + 2*lsp[i])/1000)**2) / 6) * (mom[i]/fyM))

        # Force
        if mem['bending'] == 'single':
            force = mom / (L/1000)
        else:
            force = 2 * mom / (L/1000)

        # Shear displacement
        displSh = self._computeShearDisplacement(mom, force, mn, lp, L, fycurv, displF)

        # Total displacement
        displ = displF + displSh

        # Displacement ductility
        dy1 = np.interp(fycurv, curv, displ)
        dy = (mn/fyM) * dy1
        du = displ[-1]
        displBilin = np.array([0, dy, du])
        dduct = displ / dy
        displDuctility = np.max(dduct)

        dy1f = np.interp(fycurv, curv, displF)
        dyf = (mn/fyM) * dy1f

        # Shear capacity
        self._computeShearCapacity(force, displ, displF, dyf, cMn, axialLoad, lbe, dy)

        # Store results
        self.results['plasticHingeLength'] = lp
        self.results['strainPenetration'] = lsp
        self.results['displacementFlexure'] = displF
        self.results['displacementShear'] = displSh
        self.results['displacement'] = displ
        self.results['force'] = force
        self.results['yieldDisplacement'] = dy
        self.results['ultimateDisplacement'] = du
        self.results['displacementDuctility'] = displDuctility
        self.results['displacementBilinear'] = displBilin
        if mem['bending'] == 'single':
            forceBilin = self.results['momentBilinear'] / (L/1000)
        else:
            forceBilin = 2 * self.results['momentBilinear'] / (L/1000)
        self.results['forceBilinear'] = forceBilin

    def _computeShearDisplacement(self, mom, force, mn, lp, L, fycurv, displF):
        """Calculate shear displacement component."""
        geom = self.geometryProps
        mat = self.materialProps
        mem = self.memberProps
        diameter = geom['diameter']

        # Shear modulus and properties
        G = 0.43 * mat['ec']
        As = 0.9 * self.agross
        Ig = np.pi * (diameter**4) / 64
        Ieff = (mn*1000 / (mat['ec']*1e6*self.results['equivalentCurvature'])) * 1e12

        # Reinforcement ratios
        longSteelRatio = self.ast / self.agross
        transvSteelRatio = np.pi * (geom['transBarDiam']**2) / (geom['spacing']*self.dsp)

        beta = min(0.5 + 20*longSteelRatio, 1)

        # Alpha factor
        if mem['bending'] == 'single':
            alpha = min(max(1, 3 - L/diameter), 1.5)
        else:
            alpha = min(max(1, 3 - L/(2*diameter)), 1.5)

        # Initial shear capacity
        vc1 = 0.29 * alpha * beta * 0.8 * np.sqrt(mat['fpc']) * self.agross / 1000

        # Stiffnesses
        kscr = ((0.39*transvSteelRatio) * 0.25 * mat['es'] * ((0.8*diameter/1000)**2) /
                (0.25 + 10*(0.39*transvSteelRatio))) * 1000

        if mem['bending'] == 'single':
            ksg = (G * As / L) / 1000
            kscr = kscr / L
        else:
            ksg = (G * As / (L/2)) / 1000
            kscr = kscr / (L/2)

        kseff = ksg * (Ieff / Ig)
        aux = (vc1 / kseff) / 1000

        # Compute shear displacement
        displSh = np.zeros(len(mom))
        momAux = mom.copy()
        aux2 = 0

        for i in range(len(mom)):
            if momAux[i] <= mn and force[i] < vc1:
                displSh[i] = (force[i] / kseff) / 1000
            elif momAux[i] <= mn and force[i] >= vc1:
                displSh[i] = ((force[i] - vc1) / kscr) / 1000 + aux
            else:  # momAux[i] > mn
                momAux = 4 * momAux
                aux2 += 1
                displSh[i] = (displF[i] / displF[i-1]) * displSh[i-1]

        return displSh


    def _computeShearCapacity(self, force, displ, displF, dyf, cMn, axialLoad, lbe, dy):
        """
        Compute shear capacity and check for shear failure.

        Evaluates concrete shear contribution, steel shear contribution,
        and axial load contribution to shear strength. Checks if shear
        failure occurs before flexural failure.
        """
        geom = self.geometryProps
        mat = self.materialProps
        mem = self.memberProps
        diameter = geom['diameter']
        P = axialLoad * 1000  # Convert to N

        # Steel shear contribution
        vs = (0.5 * np.pi * (0.25*np.pi*(geom['transBarDiam']**2)) * mat['fyh'] *
              np.cos(np.pi/6) * (diameter - geom['cover'] + 0.5*geom['transBarDiam'] - cMn) /
              geom['spacing']) / 1000

        vsd = (0.5 * np.pi * (0.25*np.pi*(geom['transBarDiam']**2)) * mat['fyh'] *
               np.cos(35*np.pi/180) * (diameter - geom['cover'] + 0.5*geom['transBarDiam'] - cMn) /
               geom['spacing']) / 1000

        # Parameters
        longSteelRatio = self.ast / self.agross
        beta = min(0.5 + 20*longSteelRatio, 1)
        dductF = displ / dyf

        # Alpha based on bending mode
        if mem['bending'] == 'single':
            alpha = min(max(1, 3 - mem['length']/diameter), 1.5)
            if P > 0:
                vp = (P * (diameter - cMn) / (2*mem['length'])) / 1000
            else:
                vp = 0
        else:
            alpha = min(max(1, 3 - mem['length']/(2*diameter)), 1.5)
            if P > 0:
                vp = (P * (diameter - cMn) / mem['length']) / 1000
            else:
                vp = 0

        # Concrete shear contribution (ductility-dependent)
        vc = np.zeros(len(dductF))
        if mem['ductilityMode'] == 'uniaxial':
            for i in range(len(dductF)):
                vc[i] = (alpha * beta * min(max(0.05, 0.37 - 0.04*dductF[i]), 0.29) *
                        0.8 * np.sqrt(mat['fpc']) * self.agross / 1000)
        else:  # biaxial
            for i in range(len(dductF)):
                vc[i] = (alpha * beta * min(max(0.05, 0.33 - 0.04*dductF[i]), 0.29) *
                        0.8 * np.sqrt(mat['fpc']) * self.agross / 1000)

        # Total shear capacity
        vcd = 0.862 * vc
        vpd = 0.85 * vp
        V = vc + vs + vp
        Vd = 0.85 * (vcd + vsd + vpd)

        # Check for shear failure
        shearFailure = False
        failureCriteria = 1  # 1=flexural, 2=brittle shear, 3=some ductility, 4=ductile shear
        failDispl = None
        failForce = None
        failDuct = None

        if V[-1] < force[-1]:
            failure = V - force
            failDispl = np.interp(0, failure, displ)
            failForce = np.interp(failDispl, displ, force)
            failDuct = failDispl / dy
            shearFailure = True

            # Determine failure type
            if mem['bending'] == 'single':
                if failDispl <= 2*dy:
                    failureCriteria = 2  # Brittle
                elif failDispl < 8*dy:
                    failureCriteria = 3  # Some ductility
                else:
                    failureCriteria = 4  # Ductile
            else:  # double
                if failDispl <= dy:
                    failureCriteria = 2
                elif failDispl < 7*dy:
                    failureCriteria = 3
                else:
                    failureCriteria = 4

        # Store results
        self.results['shearCapacity'] = V
        self.results['shearCapacityDesign'] = Vd
        self.results['shearConcrete'] = vc
        self.results['shearSteel'] = vs
        self.results['shearAxial'] = vp
        self.results['shearFailure'] = shearFailure
        self.results['failureCriteria'] = failureCriteria
        if shearFailure:
            self.results['shearFailureDisplacement'] = failDispl
            self.results['shearFailureForce'] = failForce
            self.results['shearFailureDuctility'] = failDuct

    def _computeBucklingModels(self):
        """
        Evaluate buckling potential using Moyer-Kowalsky and Berry-Eberhard models.
        """
        geom = self.geometryProps
        mat = self.materialProps
        mem = self.memberProps

        mcResults = self.results['momentCurvature']
        curvDuct = mcResults['curvature'] / self.results['equivalentCurvature']
        steelStrain = mcResults['steelStrain']
        displ = self.results['displacement']
        force = self.results['force']

        # Moyer-Kowalsky model
        sectionCurvDuct = self.results['sectionCurvatureDuctility']
        bucklingMK = False
        failCuDuMK = None

        if sectionCurvDuct > 4:
            esgr4 = -0.5 * np.interp(4, curvDuct, steelStrain)
            escc = 3 * ((geom['spacing']/geom['longBarDiam'])**(-2.5))

            esgr = np.zeros(len(steelStrain))
            for i in range(len(steelStrain)):
                if curvDuct[i] < 1:
                    esgr[i] = 0
                elif curvDuct[i] < 4:
                    esgr[i] = (esgr4 / 4) * curvDuct[i]
                else:
                    esgr[i] = -0.5 * steelStrain[i]

            esfl = escc - esgr

            if -steelStrain[-1] >= esfl[-1]:
                bucklingMK = True
                fail = esfl - (-steelStrain)
                failCuDuMK = np.interp(0, fail, curvDuct)

        # Berry-Eberhard model
        bucklingBE = False
        failCuDuBE = None

        # Model constants depend on axial load ratio
        axialRatio = (self.results['momentCurvature'].get('axialLoad', 0) * 1000) / (mat['fpc'] * self.agross)

        if axialRatio >= 0.30:
            C0, C1, C2, C3, C4 = 0.006, 7.190, 3.129, 0.651, 0.227
        else:
            C0, C1, C2, C3, C4 = 0.0010, 7.30, 1.30, 1.30, 3.00

        transvSteelRatio = (np.pi * (geom['transBarDiam']**2)) / (geom['spacing'] * self.dsp)
        roeff = transvSteelRatio * mat['fyh'] / mat['fpc']

        lbe = mem['length'] if mem['bending'] == 'single' else mem['length']/2
        P = 0  # Simplified - should use actual axial load

        rotb = (C0 * (1 + C1*roeff) * ((1 + C2*P/(self.agross*mat['fpc']))**(-1)) *
                (1 + C3*lbe/geom['diameter'] + C4*geom['longBarDiam']*mat['fy']/geom['diameter']))

        fycurv = self.results['yieldCurvature']
        lp = self.results['plasticHingeLength']
        plrot = (mcResults['curvature'] - fycurv) * (lp / 1000)

        if np.max(plrot) > rotb:
            bucklingBE = True
            failBE = plrot - rotb
            failCuDuBE = np.interp(0, failBE, curvDuct)

        # Store results
        self.results['bucklingMoyerKowalsky'] = bucklingMK
        if bucklingMK:
            self.results['bucklingMKCurvatureDuctility'] = failCuDuMK
        self.results['bucklingBerryEberhard'] = bucklingBE
        if bucklingBE:
            self.results['bucklingBECurvatureDuctility'] = failCuDuBE

    def _computeLimitStates(self):
        """
        Evaluate deformation limit states (serviceability and damage control).
        """
        mcResults = self.results['momentCurvature']
        lim = self.limitStates

        coverStrain = mcResults['coverStrain']
        steelStrain = mcResults['steelStrain']
        displ = self.results['displacement']
        curv = mcResults['curvature']
        mom = mcResults['moment']
        force = self.results['force']
        curvDuct = curv / self.results['equivalentCurvature']
        displDuct = displ / self.results['yieldDisplacement']

        # Initialize
        limitStateResults = {
            'serviceability': {},
            'damageControl': {},
            'ultimate': {}
        }

        # Serviceability
        if np.max(coverStrain) > lim['ecser'] or np.max(np.abs(steelStrain)) > abs(lim['esser']):
            displSerC = np.interp(lim['ecser'], coverStrain, displ)
            displSerS = np.interp(lim['esser'], steelStrain, displ)
            displSer = min(displSerC, displSerS)

            limitStateResults['serviceability'] = {
                'displacement': displSer,
                'displacementDuctility': np.interp(displSer, displ, displDuct),
                'curvature': np.interp(displSer, displ, curv),
                'curvatureDuctility': np.interp(displSer, displ, curvDuct),
                'coverStrain': np.interp(displSer, displ, coverStrain),
                'steelStrain': np.interp(displSer, displ, steelStrain),
                'moment': np.interp(displSer, displ, mom),
                'force': np.interp(displSer, displ, force)
            }

            # Damage control
            if np.max(coverStrain) > lim['ecdam'] or np.max(np.abs(steelStrain)) > abs(lim['esdam']):
                displDamC = np.interp(lim['ecdam'], coverStrain, displ)
                displDamS = np.interp(lim['esdam'], steelStrain, displ)
                displDam = min(displDamC, displDamS)

                limitStateResults['damageControl'] = {
                    'displacement': displDam,
                    'displacementDuctility': np.interp(displDam, displ, displDuct),
                    'curvature': np.interp(displDam, displ, curv),
                    'curvatureDuctility': np.interp(displDam, displ, curvDuct),
                    'coverStrain': np.interp(displDam, displ, coverStrain),
                    'steelStrain': np.interp(displDam, displ, steelStrain),
                    'moment': np.interp(displDam, displ, mom),
                    'force': np.interp(displDam, displ, force)
                }

        # Ultimate
        limitStateResults['ultimate'] = {
            'displacement': displ[-1],
            'displacementDuctility': displDuct[-1],
            'curvature': curv[-1],
            'curvatureDuctility': curvDuct[-1],
            'coverStrain': coverStrain[-1],
            'steelStrain': steelStrain[-1],
            'moment': mom[-1],
            'force': force[-1]
        }

        self.results['limitStates'] = limitStateResults

    def _computeInteractionDiagram(self):
        """
        Compute axial load-moment (P-M) interaction diagram.

        Computes the interaction surface by analyzing the section at multiple
        axial load levels from pure tension to pure compression.

        Results stored in self.results['interactionDiagram'] containing:
            - axialLoads: Array of P values (kN)
            - moments: Array of M values at each P (kN-m)
            - axialLoadsApprox: Simplified bilinear approximation points
            - momentsApprox: Simplified moments for bilinear approximation
        """
        geom = self.geometryProps
        mat = self.materialProps
        lim = self.limitStates

        # Calculate yield surface forces
        if hasattr(self, 'dsp'):  # Circular
            acore = 0.25 * np.pi * (self.dsp**2)
            diameter = geom['diameter']
        else:  # Rectangular
            acore = self.hcore * self.bcore
            diameter = geom['height']  # Use height as characteristic dimension

        # Pure compression and tension forces for yield surface
        pCid = (self._interpolateStress(lim['csid'], self.strainConf, self.stressConf) *
                (acore - self.ast) +
                self._interpolateStress(lim['csid'], self.strainUnconf, self.stressUnconf) *
                (self.agross - acore) +
                self.ast * self._interpolateStress(lim['csid'], self.strainSteel, self.stressSteel))

        pTid = self.ast * self._interpolateStress(lim['ssid'], self.strainSteel, self.stressSteel)

        # Create axial load vector
        # From 90% tension to 70% compression
        axialLoads = np.concatenate([
            np.arange(-0.90*pTid, 0, 0.30*pTid),
            np.arange(0.05*mat['fpc']*self.agross, 0.7*pCid, 0.05*mat['fpc']*self.agross)
        ])

        nPoints = len(axialLoads)
        moments = np.zeros(nPoints)
        concreteStrains = np.zeros(nPoints)
        steelStrains = np.zeros(nPoints)

        print(f"  Computing {nPoints} points on P-M interaction curve...")

        # Compute moment capacity at each axial load
        for i in range(nPoints):
            P = axialLoads[i]

            # Perform moment-curvature analysis at this axial load
            try:
                # Temporarily modify setup to compute for this P
                mcResults = self._computeMomentCurvature(P / 1000)  # Convert to kN

                # Extract nominal moment at limit state strains
                coverStrain = mcResults['coverStrain']
                steelStrain = mcResults['steelStrain']
                mom = mcResults['moment']

                # Nominal moment based on serviceability criteria
                mn = np.interp(lim['csid'], coverStrain, mom, left=np.nan, right=np.nan)
                esaux = np.interp(lim['csid'], coverStrain, steelStrain, left=np.nan, right=np.nan)

                # Check if steel controls
                if abs(esaux) > abs(lim['ssid']) or np.isnan(mn):
                    mnSteel = np.interp(-lim['ssid'], steelStrain, mom, left=np.nan, right=np.nan)
                    if not np.isnan(mnSteel):
                        mn = mnSteel
                        concreteStrains[i] = np.interp(-lim['ssid'], steelStrain, coverStrain)
                        steelStrains[i] = -lim['ssid']
                    else:
                        # Use maximum moment if nominal cannot be determined
                        mn = np.max(mom)
                        maxIdx = np.argmax(mom)
                        concreteStrains[i] = coverStrain[maxIdx]
                        steelStrains[i] = steelStrain[maxIdx]
                else:
                    concreteStrains[i] = lim['csid']
                    steelStrains[i] = esaux

                moments[i] = mn

            except Exception as e:
                print(f"  Warning: Failed to compute point {i+1}/{nPoints} at P={P/1000:.1f} kN")
                moments[i] = 0

            # Progress indicator
            if (i + 1) % max(1, nPoints // 10) == 0:
                print(f"  Progress: {i+1}/{nPoints} points complete")

        # Add end points (pure compression and pure tension)
        axialLoadsComplete = np.concatenate([[-pTid], axialLoads, [pCid]])
        momentsComplete = np.concatenate([[0], moments, [0]])

        # Create simplified bilinear approximation for NLTHA
        # Find balanced point (maximum moment)
        maxMomentIdx = np.argmax(momentsComplete)
        pB = axialLoadsComplete[maxMomentIdx]
        mB = momentsComplete[maxMomentIdx]

        # Points at 1/3 and 2/3 of balanced load
        pB13 = pB / 3
        mB13 = np.interp(pB13, axialLoadsComplete, momentsComplete)

        pB23 = 2 * pB / 3
        mB23 = np.interp(pB23, axialLoadsComplete, momentsComplete)

        # Moment at zero axial load
        mB0 = np.interp(0, axialLoadsComplete, momentsComplete)

        # Simplified points for NLTHA
        axialLoadsApprox = np.array([-pTid, 0, pB13, pB23, pB, pCid])
        momentsApprox = np.array([0, mB0, mB13, mB23, mB, 0])

        # Store results
        self.results['interactionDiagram'] = {
            'axialLoads': axialLoadsComplete / 1000,  # Convert to kN
            'moments': momentsComplete,
            'concreteStrains': concreteStrains,
            'steelStrains': steelStrains,
            'axialLoadsApprox': axialLoadsApprox / 1000,  # kN
            'momentsApprox': momentsApprox,
            'balancedPoint': {'P': pB / 1000, 'M': mB},
            'pureTension': -pTid / 1000,
            'pureCompression': pCid / 1000
        }

        print(f"  P-M interaction complete!")
        print(f"    Pure tension: {-pTid/1000:.1f} kN")
        print(f"    Pure compression: {pCid/1000:.1f} kN")
        print(f"    Balanced point: P={pB/1000:.1f} kN, M={mB:.1f} kN-m")

    def plotResults(self):
        """
        Create comprehensive plots of analysis results.

        Generates plots for:
        - Material models
        - Moment-curvature
        - Force-displacement
        - Limit states
        - Buckling models (if applicable)
        """
        # Material models
        self.plotMaterialModels()

        # Moment-curvature
        fig, ax = plt.subplots(figsize=(10, 6))
        mcRes = self.results['momentCurvature']
        ax.plot(mcRes['curvature'], mcRes['moment'], 'b-', linewidth=2,
                label='M-φ Response')
        ax.plot(self.results['curvatureBilinear'], self.results['momentBilinear'],
                'r--', linewidth=2, label='Bilinear Approximation')
        ax.set_xlabel('Curvature (1/m)', fontsize=14)
        ax.set_ylabel('Moment (kN-m)', fontsize=14)
        ax.set_title('Moment-Curvature Relation', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Force-displacement
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.results['displacement'], self.results['force'], 'k-',
                linewidth=2, label='Total Response')
        ax.plot(self.results['displacementBilinear'], self.results['forceBilinear'],
                'b--', linewidth=2, label='Bilinear Approximation')
        ax.plot(self.results['displacement'], self.results['shearCapacity'], 'r:',
                linewidth=2, label='Shear Capacity (Assessment)')
        ax.plot(self.results['displacement'], self.results['shearCapacityDesign'], 'm:',
                linewidth=2, label='Shear Capacity (Design)')

        # Mark failure points if any
        if self.results.get('shearFailure'):
            ax.plot(self.results['shearFailureDisplacement'],
                   self.results['shearFailureForce'], 'mo', markersize=10,
                   markerfacecolor='g', markeredgecolor='k',
                   label='Shear Failure')

        ax.set_xlabel('Displacement (m)', fontsize=14)
        ax.set_ylabel('Force (kN)', fontsize=14)
        ax.set_title('Force-Displacement Relation', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plotInteractionDiagram(self):
        """
        Plot P-M interaction diagram if it has been computed.

        Creates a plot showing:
        - Full interaction curve
        - Simplified bilinear approximation
        - Balanced point
        - Pure compression and tension limits
        """
        if 'interactionDiagram' not in self.results:
            print("P-M interaction diagram not computed. Run analyze(performInteraction=True) first.")
            return

        interaction = self.results['interactionDiagram']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot full interaction curve
        ax.plot(interaction['moments'], interaction['axialLoads'], 'b-',
                linewidth=2.5, label='P-M Interaction Curve')

        # Plot simplified approximation
        ax.plot(interaction['momentsApprox'], interaction['axialLoadsApprox'], 'r--',
                linewidth=2, marker='o', markersize=8,
                label='Simplified Approximation')

        # Mark balanced point
        bp = interaction['balancedPoint']
        ax.plot(bp['M'], bp['P'], 'go', markersize=12,
                markerfacecolor='lime', markeredgecolor='darkgreen',
                markeredgewidth=2, label=f"Balanced Point\n(P={bp['P']:.1f} kN, M={bp['M']:.1f} kN-m)")

        # Mark pure compression and tension
        ax.axhline(y=interaction['pureCompression'], color='gray',
                  linestyle=':', linewidth=1.5, label=f"Pure Compression ({interaction['pureCompression']:.1f} kN)")
        ax.axhline(y=interaction['pureTension'], color='gray',
                  linestyle=':', linewidth=1.5, label=f"Pure Tension ({interaction['pureTension']:.1f} kN)")

        # Mark zero axial load
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Moment (kN-m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Axial Load (kN)', fontsize=14, fontweight='bold')
        ax.set_title('P-M Interaction Diagram', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

    def writeResults(self, filename):
        """
        Write analysis results to a text file.

        Parameters:
            filename : str
                Output filename
        """
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" CUMBIA CIRCULAR SECTION ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")

            # Section geometry
            f.write("SECTION GEOMETRY:\n")
            f.write(f"  Diameter: {self.geometryProps['diameter']:.1f} mm\n")
            f.write(f"  Cover: {self.geometryProps['cover']:.1f} mm\n")
            f.write(f"  Number of longitudinal bars: {self.geometryProps['numLongBars']}\n")
            f.write(f"  Longitudinal bar diameter: {self.geometryProps['longBarDiam']:.1f} mm\n")
            f.write(f"  Transverse bar diameter: {self.geometryProps['transBarDiam']:.1f} mm\n")
            f.write(f"  Spacing: {self.geometryProps['spacing']:.1f} mm\n\n")

            # Material properties
            f.write("MATERIAL PROPERTIES:\n")
            f.write(f"  Concrete strength (f'c): {self.materialProps['fpc']:.1f} MPa\n")
            f.write(f"  Steel yield strength (fy): {self.materialProps['fy']:.1f} MPa\n")
            f.write(f"  Steel ultimate strength (fsu): {self.materialProps['fsu']:.1f} MPa\n\n")

            # Key results
            f.write("SECTION CAPACITY:\n")
            f.write(f"  Nominal moment: {self.results['nominalMoment']:.2f} kN-m\n")
            f.write(f"  Yield moment: {self.results['yieldMoment']:.2f} kN-m\n")
            f.write(f"  Section curvature ductility: {self.results['sectionCurvatureDuctility']:.2f}\n")
            f.write(f"  Displacement ductility: {self.results['displacementDuctility']:.2f}\n\n")

            # Failure mode
            f.write("FAILURE MODE:\n")
            criteria = self.results['failureCriteria']
            if criteria == 1:
                f.write("  Flexural failure\n")
            elif criteria == 2:
                f.write("  Brittle shear failure\n")
            elif criteria == 3:
                f.write("  Shear failure at some ductility\n")
            elif criteria == 4:
                f.write("  Ductile shear failure\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Analysis completed successfully\n")

        print(f"Results written to {filename}")


# =============================================================================
# RECTANGULAR SECTION CLASS
# =============================================================================

class RectangularSection(RCSection):
    """
    Rectangular reinforced concrete section analysis.

    This class performs comprehensive analysis of rectangular RC members including:
        - Moment-curvature analysis
        - Force-displacement analysis
        - P-M interaction diagrams
        - Buckling assessment
        - Shear capacity evaluation
        - Deformation limit states

    Example:
        >>> section = RectangularSection(
        ...     height=400,
        ...     width=300,
        ...     cover=40,
        ...     reinforcementLayers=[
        ...         [52.7, 3, 25.4],  # [distance from top, num bars, diameter]
        ...         [347.3, 3, 25.4]
        ...     ],
        ...     transBarDiam=9.5,
        ...     spacing=120,
        ...     numLegsX=2,
        ...     numLegsY=2
        ... )
        >>> section.setMaterialProperties(fpc=28, fy=450, fyh=400)
        >>> section.setMemberProperties(length=1200, bending='single')
        >>> section.analyze(axialLoad=-200)
    """

    def __init__(self, height, width, cover, reinforcementLayers,
                 transBarDiam, spacing, numLegsX=2, numLegsY=2):
        """
        Initialize rectangular section geometry.

        Parameters:
            height : float
                Section height (mm)
            width : float
                Section width (mm)
            cover : float
                Cover to longitudinal bars (mm)
            reinforcementLayers : list of lists
                Each row: [distance from top (mm), number of bars, bar diameter (mm)]
            transBarDiam : float
                Transverse reinforcement diameter (mm)
            spacing : float
                Spacing of transverse steel (mm)
            numLegsX : int, optional
                Number of legs of transverse steel in X direction (confinement)
            numLegsY : int, optional
                Number of legs of transverse steel in Y direction (shear)
        """
        super().__init__()

        # Store geometry
        self.geometryProps = {
            'height': height,
            'width': width,
            'cover': cover,
            'reinforcementLayers': np.array(reinforcementLayers),
            'transBarDiam': transBarDiam,
            'spacing': spacing,
            'numLegsX': numLegsX,
            'numLegsY': numLegsY
        }

        # Calculate derived geometry
        self.hcore = height - 2*cover + transBarDiam
        self.bcore = width - 2*cover + transBarDiam
        self.dcore = cover - transBarDiam*0.5

        # Calculate total steel area
        self.ast = 0
        for layer in reinforcementLayers:
            self.ast += layer[1] * 0.25 * np.pi * (layer[2]**2)

        self.agross = height * width

    def setMaterialProperties(self, fpc, fy, fyh, es=200000, fsu=None,
                             esh=0.008, esu=0.12, eco=0.002, esm=0.11,
                             espall=0.0064, ec=None, c1=3.5, eyPlateau=350,
                             concreteModel='mc', steelModel='ra',
                             lightweightConcrete=False):
        """Set material properties (same interface as CircularSection)."""
        if fsu is None:
            fsu = 1.35 * fy
        if ec is None:
            ec = 5000 * np.sqrt(fpc)

        self.materialProps = {
            'fpc': fpc, 'fy': fy, 'fyh': fyh, 'es': es, 'fsu': fsu,
            'esh': esh, 'esu': esu, 'eco': eco, 'esm': esm, 'espall': espall,
            'ec': ec, 'c1': c1, 'eyPlateau': eyPlateau,
            'concreteModel': concreteModel, 'steelModel': steelModel,
            'lightweightConcrete': lightweightConcrete
        }

        self._generateMaterialModels()

    def _generateMaterialModels(self):
        """Generate stress-strain curves for rectangular section."""
        geom = self.geometryProps
        mat = self.materialProps

        # Calculate clear distances between bars for confinement
        wi = self._calculateClearDistances()

        # Confined concrete
        if mat['concreteModel'].lower() == 'mc':
            if mat['lightweightConcrete']:
                self.strainConf, self.stressConf = MaterialModels.manderConfinedLightweight(
                    mat['ec'], self.ast, geom['transBarDiam'], geom['cover'],
                    geom['spacing'], mat['fpc'], mat['fyh'], mat['eco'], mat['esm'],
                    mat['espall'], 'rectangular', height=geom['height'], width=geom['width'],
                    ncx=geom['numLegsX'], ncy=geom['numLegsY'], wi=wi,
                    dels=self.analysisParams['deltaStrain'], transType='hoops'
                )
            else:
                self.strainConf, self.stressConf = MaterialModels.manderConfined(
                    mat['ec'], self.ast, geom['transBarDiam'], geom['cover'],
                    geom['spacing'], mat['fpc'], mat['fyh'], mat['eco'], mat['esm'],
                    mat['espall'], 'rectangular', height=geom['height'], width=geom['width'],
                    ncx=geom['numLegsX'], ncy=geom['numLegsY'], wi=wi,
                    dels=self.analysisParams['deltaStrain'], transType='hoops'
                )
        else:
            raise NotImplementedError("Custom material models not yet implemented")

        # Unconfined concrete
        if mat['lightweightConcrete']:
            raise NotImplementedError("Lightweight unconfined for rectangular not implemented")
        else:
            self.strainUnconf, self.stressUnconf = MaterialModels.manderUnconfined(
                mat['ec'], self.ast, geom['transBarDiam'], geom['cover'],
                geom['spacing'], mat['fpc'], mat['fyh'], mat['eco'], mat['esm'],
                mat['espall'], 'rectangular', height=geom['height'], width=geom['width'],
                ncx=geom['numLegsX'], ncy=geom['numLegsY'], wi=wi,
                dels=self.analysisParams['deltaStrain']
            )

        # Steel model
        if mat['steelModel'].lower() == 'ra':
            self.strainSteel, self.stressSteel = MaterialModels.steelRaynor(
                mat['es'], mat['fy'], mat['fsu'], mat['esh'], mat['esu'],
                dels=self.analysisParams['deltaStrain'], c1=mat['c1'],
                ey_plateau=mat['eyPlateau']
            )
        elif mat['steelModel'].lower() == 'ks':
            self.strainSteel, self.stressSteel = MaterialModels.steelKing(
                mat['es'], mat['fy'], mat['fsu'], mat['esh'], mat['esu'],
                dels=self.analysisParams['deltaStrain']
            )
        else:
            raise ValueError("Steel model must be 'ra' or 'ks'")

        # Extend curves
        self.ecu = self.strainConf[-1]
        self.ecuMander = self.ecu / 1.5
        dels = self.analysisParams['deltaStrain']

        self.strainConf = np.concatenate([[-1e10], self.strainConf,
                                          [self.strainConf[-1] + dels, 1e10]])
        self.stressConf = np.concatenate([[0], self.stressConf, [0, 0]])

        self.strainUnconf = np.concatenate([[-1e10], self.strainUnconf,
                                            [self.strainUnconf[-1] + dels, 1e10]])
        self.stressUnconf = np.concatenate([[0], self.stressUnconf, [0, 0]])

        self.esu = self.strainSteel[-1]
        self.strainSteel = np.concatenate([self.strainSteel,
                                           [self.strainSteel[-1] + dels, 1e10]])
        self.stressSteel = np.concatenate([self.stressSteel, [0, 0]])

        strainSteelNeg = -self.strainSteel[::-1]
        stressSteelNeg = -self.stressSteel[::-1]
        self.strainSteel = np.concatenate([strainSteelNeg, self.strainSteel[1:]])
        self.stressSteel = np.concatenate([stressSteelNeg, self.stressSteel[1:]])

    def _calculateClearDistances(self):
        """Calculate clear distances between peripheral bars."""
        geom = self.geometryProps
        layers = geom['reinforcementLayers']

        if len(layers) == 1:
            # Single layer
            return np.array([(geom['height'] - 2*geom['cover'] - 2*layers[0,2])*2,
                           (geom['width'] - 2*geom['cover'] - 2*layers[0,2])*2])

        # Multiple layers - calculate spacing
        wi = []
        # Top layer
        if layers[0,1] > 1:
            spacing_top = (geom['width'] - 2*geom['cover'] - layers[0,1]*layers[0,2]) / (layers[0,1] - 1)
            wi.extend([spacing_top] * int(layers[0,1] - 1))

        # Bottom layer
        if layers[-1,1] > 1:
            spacing_bot = (geom['width'] - 2*geom['cover'] - layers[-1,1]*layers[-1,2]) / (layers[-1,1] - 1)
            wi.extend([spacing_bot] * int(layers[-1,1] - 1))

        # Vertical spacing between layers
        for i in range(len(layers)-1):
            vert_spacing = layers[i+1,0] - layers[i,0] - 0.5*(layers[i,2] + layers[i+1,2])
            wi.extend([vert_spacing] * 2)

        return np.array(wi)

    def setMemberProperties(self, length, bending='single', ductilityMode='uniaxial',
                           temperature=40, kLsp=0.022):
        """Set member-level properties."""
        self.memberProps = {
            'length': length,
            'bending': bending.lower(),
            'ductilityMode': ductilityMode.lower(),
            'temperature': temperature,
            'kLsp': kLsp
        }

    def setLimitStates(self, concreteServiceStrain=0.004, steelServiceStrain=0.015,
                      concreteDamageStrain=0.018, steelDamageStrain=0.060,
                      concreteInteractionStrain=0.004, steelInteractionStrain=0.015):
        """Set deformation limit state criteria."""
        if isinstance(concreteDamageStrain, str) and concreteDamageStrain.lower() == 'twth':
            concreteDamageStrain = self.ecuMander

        self.limitStates = {
            'ecser': concreteServiceStrain,
            'esser': -steelServiceStrain,
            'ecdam': concreteDamageStrain,
            'esdam': -steelDamageStrain,
            'csid': concreteInteractionStrain,
            'ssid': steelInteractionStrain
        }

    def _setupConcreteLayers(self):
        """Setup concrete layers for rectangular section."""
        geom = self.geometryProps
        height = geom['height']
        width = geom['width']
        ncl = self.analysisParams['numConcreteLayers']

        # Layer boundaries
        tcl = height / ncl
        yl = np.arange(tcl, height + tcl, tcl)
        yl = np.sort(np.concatenate([yl, [self.dcore, height - self.dcore]]))
        yl = np.unique(yl)

        # Total layer areas
        atc = np.diff(np.concatenate([[0], yl])) * width

        # Confined layer areas
        yc = yl - self.dcore
        yc = yc[(yc > 0) & (yc < self.hcore)]
        yc = np.append(yc, self.hcore)
        atcc = np.diff(np.concatenate([[0], yc])) * self.bcore

        # Assign to confined/unconfined
        conclay = np.zeros((len(yl), 2))
        k = 0
        for i in range(len(yl)):
            if yl[i] <= self.dcore or yl[i] > height - self.dcore:
                conclay[i, :] = [atc[i], 0]
            else:
                conclay[i, :] = [atc[i] - atcc[k], atcc[k]]
                k += 1

        # Centroids
        centroids = np.concatenate([[yl[0]/2], 0.5*(yl[:-1] + yl[1:])])

        self.concreteLayers = np.column_stack([
            centroids, conclay[:, 0], conclay[:, 1], yl, np.zeros(len(yl))
        ])

    def _setupRebars(self):
        """Setup longitudinal reinforcement."""
        geom = self.geometryProps
        layers = geom['reinforcementLayers']

        # Build arrays of bar positions and areas
        distld = []
        areas = []
        diameters = []

        for layer in layers:
            dist, nBars, diam = layer
            for _ in range(int(nBars)):
                distld.append(dist)
                areas.append(0.25 * np.pi * (diam**2))
                diameters.append(diam)

        self.rebarDist = np.array(sorted(distld))
        self.rebarAreas = np.array([areas[i] for i in np.argsort(distld)])
        self.rebarDiams = np.array([diameters[i] for i in np.argsort(distld)])

        # Assign to layers and correct concrete areas
        for k in range(1, len(self.concreteLayers) - 1):
            rebarInLayer = ((self.rebarDist <= self.concreteLayers[k, 3]) &
                           (self.rebarDist > self.concreteLayers[k-1, 3]))
            self.concreteLayers[k, 4] = np.sum(self.rebarAreas[rebarInLayer])

            if self.concreteLayers[k, 2] == 0:
                self.concreteLayers[k, 1] -= self.concreteLayers[k, 4]
                if self.concreteLayers[k, 1] < 0:
                    raise ValueError("Negative concrete area")
            else:
                self.concreteLayers[k, 2] -= self.concreteLayers[k, 4]
                if self.concreteLayers[k, 2] < 0:
                    raise ValueError("Negative concrete area")

    def _computeMomentCurvature(self, axialLoad):
        """Compute M-φ for rectangular section (same algorithm as circular)."""
        geom = self.geometryProps
        mat = self.materialProps
        height = geom['height']
        P = axialLoad * 1000

        deformation = self._createDeformationVector(self.ecu)

        # Adjust for compression
        if P > 0:
            newDef = []
            for defVal in deformation:
                stressUnconf = self._interpolateStress(
                    defVal * np.ones(len(self.concreteLayers)),
                    self.strainUnconf, self.stressUnconf)
                stressConf = self._interpolateStress(
                    defVal * np.ones(len(self.concreteLayers)),
                    self.strainConf, self.stressConf)
                stressSteel = self._interpolateStress(
                    defVal * np.ones(len(self.rebarDist)),
                    self.strainSteel, self.stressSteel)

                compCheck = (np.sum(stressUnconf * self.concreteLayers[:, 1]) +
                            np.sum(stressConf * self.concreteLayers[:, 2]) +
                            np.sum(self.rebarAreas * stressSteel))

                if compCheck >= P:
                    newDef.append(defVal)
            deformation = np.array(newDef)

        np_points = len(deformation)
        curvature = np.zeros(np_points + 1)
        moment = np.zeros(np_points + 1)
        neutralAxis = np.zeros(np_points + 1)
        forceError = np.zeros(np_points + 1)
        numIter = np.zeros(np_points + 1, dtype=int)
        coverStrain = np.zeros(np_points + 1)
        coreStrain = np.zeros(np_points + 1)
        steelStrain = np.zeros(np_points + 1)

        tol = self.analysisParams['tolerance'] * height * geom['width'] * mat['fpc']
        iterMax = self.analysisParams['iterMax']
        x = height / 2
        message = 0

        for k in range(np_points):
            lostMomControl = np.max(moment[:k+1])
            if k > 0 and moment[k] < 0.8 * lostMomControl:
                message = 4
                break

            F = 10 * tol
            niter = 0

            while abs(F) > tol:
                niter += 1

                if x <= height:
                    strainConcrete = (deformation[k] / x) * (
                        self.concreteLayers[:, 0] - (height - x))
                    strainRebar = (deformation[k] / x) * (
                        self.rebarDist - (height - x))
                else:
                    strainConcrete = (deformation[k] / x) * (
                        x - height + self.concreteLayers[:, 0])
                    strainRebar = (deformation[k] / x) * (
                        x - height + self.rebarDist)

                stressUnconf = self._interpolateStress(strainConcrete,
                                                       self.strainUnconf, self.stressUnconf)
                stressConf = self._interpolateStress(strainConcrete,
                                                     self.strainConf, self.stressConf)
                stressRebar = self._interpolateStress(strainRebar,
                                                      self.strainSteel, self.stressSteel)

                forceUnconf = stressUnconf * self.concreteLayers[:, 1]
                forceConf = stressConf * self.concreteLayers[:, 2]
                forceSteel = self.rebarAreas * stressRebar

                F = np.sum(forceUnconf) + np.sum(forceConf) + np.sum(forceSteel) - P

                if F > 0:
                    x = x - 0.05 * x
                else:
                    x = x + 0.05 * x

                if niter > iterMax:
                    message = 3
                    break

            if message == 3:
                break

            cores = (deformation[k] / x) * abs(x - self.dcore)
            sameModel = (mat['concreteModel'] == 'mu')

            if not sameModel:
                if cores >= self.ecu:
                    message = 1
                    break
            else:
                if deformation[k] >= self.ecu:
                    message = 1
                    break

            if abs(strainRebar[0]) > self.esu:
                message = 2
                break

            neutralAxis[k+1] = x
            forceError[k+1] = F
            numIter[k+1] = niter

            momentVal = (np.sum(forceUnconf * self.concreteLayers[:, 0]) +
                        np.sum(forceConf * self.concreteLayers[:, 0]) +
                        np.sum(forceSteel * self.rebarDist) -
                        P * (height / 2)) / 1e6

            if momentVal < 0:
                momentVal = -0.01 * momentVal

            moment[k+1] = momentVal
            curvature[k+1] = 1000 * deformation[k] / x
            coverStrain[k+1] = deformation[k]
            coreStrain[k+1] = cores
            steelStrain[k+1] = strainRebar[0]

            if message != 0:
                break

        lastIdx = k + 2 if message != 0 else np_points + 1

        return {
            'curvature': curvature[:lastIdx],
            'moment': moment[:lastIdx],
            'neutralAxis': neutralAxis[:lastIdx],
            'coverStrain': coverStrain[:lastIdx],
            'coreStrain': coreStrain[:lastIdx],
            'steelStrain': steelStrain[:lastIdx],
            'forceError': forceError[:lastIdx],
            'numIterations': numIter[:lastIdx],
            'message': message
        }

    # Use same analyze() workflow as CircularSection
    def analyze(self, axialLoad=0, performInteraction=False):
        """Perform complete analysis (same as CircularSection)."""
        if not hasattr(self, 'memberProps'):
            raise ValueError("Member properties not set")
        if not hasattr(self, 'limitStates'):
            self.setLimitStates()

        print("Setting up fiber section...")
        self._setupConcreteLayers()
        self._setupRebars()

        print(f"Computing moment-curvature for P = {axialLoad} kN...")
        mcResults = self._computeMomentCurvature(axialLoad)
        self.results['momentCurvature'] = mcResults

        print("Computing section properties...")
        self._computeSectionProperties(mcResults, axialLoad)

        print("Computing member force-displacement...")
        self._computeForceDisplacement(mcResults, axialLoad)

        print("Evaluating buckling models...")
        self._computeBucklingModels()

        print("Evaluating limit states...")
        self._computeLimitStates()

        if performInteraction:
            print("Computing P-M interaction diagram...")
            self._computeInteractionDiagram()

        print("Analysis complete!")
        return self.results

    # Reuse methods from CircularSection (they work for both)
    _computeSectionProperties = CircularSection._computeSectionProperties
    _computeForceDisplacement = CircularSection._computeForceDisplacement
    _computeShearDisplacement = CircularSection._computeShearDisplacement
    _computeShearCapacity = CircularSection._computeShearCapacity
    _computeBucklingModels = CircularSection._computeBucklingModels
    _computeLimitStates = CircularSection._computeLimitStates
    _computeInteractionDiagram = CircularSection._computeInteractionDiagram
    plotResults = CircularSection.plotResults
    plotInteractionDiagram = CircularSection.plotInteractionDiagram
    writeResults = CircularSection.writeResults


# =============================================================================
# TEST EXAMPLES
# =============================================================================

def test_circular_section():
    """
    Complete test example for circular RC section analysis.

    This example demonstrates:
    - Circular section setup
    - Material model generation
    - Moment-curvature analysis
    - Force-displacement analysis
    - P-M interaction diagram
    - Result plotting and export
    """
    print("="*80)
    print(" TEST 1: CIRCULAR SECTION ANALYSIS")
    print("="*80)
    print()
    print("Section: D=1000mm circular column")
    print("Reinforcement: 22-Φ25mm bars, spirals Φ9mm @ 120mm")
    print("Materials: f'c=35 MPa, fy=460 MPa")
    print("Member: L=3000mm, single curvature, P=400 kN compression")
    print("="*80)
    print()

    # Create circular section
    section = CircularSection(
        diameter=1000,          # mm
        cover=50,               # mm
        numLongBars=22,
        longBarDiam=25,         # mm
        transBarDiam=9,         # mm
        spacing=120,            # mm
        transType='spirals'
    )

    # Set material properties
    section.setMaterialProperties(
        fpc=35,                 # MPa
        fy=460,                 # MPa
        fyh=400,                # MPa
        fsu=620,                # MPa
        concreteModel='mc',     # Mander confined
        steelModel='ra'         # Raynor
    )

    # Set member properties
    section.setMemberProperties(
        length=3000,            # mm
        bending='single',
        ductilityMode='biaxial'
    )

    # Set limit states
    section.setLimitStates(
        concreteServiceStrain=0.004,
        steelServiceStrain=0.015,
        concreteDamageStrain=0.018,
        steelDamageStrain=0.060
    )

    # Perform analysis with P-M interaction
    print("Starting analysis...")
    results = section.analyze(axialLoad=400, performInteraction=True)

    # Print key results
    print("\n" + "="*80)
    print("KEY RESULTS - CIRCULAR SECTION:")
    print("="*80)
    print(f"Nominal Moment:              {results['nominalMoment']:.2f} kN-m")
    print(f"Yield Moment:                {results['yieldMoment']:.2f} kN-m")
    print(f"Ultimate Moment:             {results['momentCurvature']['moment'][-1]:.2f} kN-m")
    print(f"Yield Curvature:             {results['yieldCurvature']:.5f} 1/m")
    print(f"Ultimate Curvature:          {results['momentCurvature']['curvature'][-1]:.5f} 1/m")
    print(f"Section Curvature Ductility: {results['sectionCurvatureDuctility']:.2f}")
    print(f"Yield Displacement:          {results['yieldDisplacement']*1000:.2f} mm")
    print(f"Ultimate Displacement:       {results['ultimateDisplacement']*1000:.2f} mm")
    print(f"Displacement Ductility:      {results['displacementDuctility']:.2f}")
    print(f"Plastic Hinge Length:        {results['plasticHingeLength']:.1f} mm")
    print()
    print("Shear Capacity:")
    print(f"  Concrete contribution:     {results['shearConcrete'][-1]:.2f} kN")
    print(f"  Steel contribution:        {results['shearSteel']:.2f} kN")
    print(f"  Total capacity:            {results['shearCapacity'][-1]:.2f} kN")
    print(f"  Shear failure:             {'YES' if results['shearFailure'] else 'NO'}")
    print()
    print("Buckling:")
    print(f"  Moyer-Kowalsky buckling:   {'YES' if results['bucklingMoyerKowalsky'] else 'NO'}")
    print(f"  Berry-Eberhard buckling:   {'YES' if results['bucklingBerryEberhard'] else 'NO'}")
    print()

    if 'interactionDiagram' in results:
        interaction = results['interactionDiagram']
        print("P-M Interaction:")
        print(f"  Pure tension capacity:     {interaction['pureTension']:.1f} kN")
        print(f"  Pure compression capacity: {interaction['pureCompression']:.1f} kN")
        print(f"  Balanced point:            P={interaction['balancedPoint']['P']:.1f} kN, M={interaction['balancedPoint']['M']:.1f} kN-m")

    print("="*80)

    # Write detailed results
    section.writeResults('CUMBIA_circular_results.txt')
    print("\nDetailed results written to 'CUMBIA_circular_results.txt'")

    # Plot results
    print("\nGenerating plots...")
    section.plotMaterialModels()
    section.plotResults()

    if 'interactionDiagram' in results:
        section.plotInteractionDiagram()

    print("\n" + "="*80)
    print("CIRCULAR SECTION TEST COMPLETE!")
    print("="*80)
    print()

    return section


def test_rectangular_section():
    """
    Complete test example for rectangular RC section analysis.

    This example demonstrates:
    - Rectangular section with multiple reinforcement layers
    - Material model generation for rectangular geometry
    - Complete M-φ and F-Δ analysis
    - P-M interaction diagram
    - Comparison with circular section behavior
    """
    print("="*80)
    print(" TEST 2: RECTANGULAR SECTION ANALYSIS")
    print("="*80)
    print()
    print("Section: 400x400mm square column")
    print("Reinforcement: 3-Φ25mm top + 3-Φ25mm bottom, hoops Φ10mm @ 150mm")
    print("Materials: f'c=28 MPa, fy=450 MPa")
    print("Member: L=1200mm, single curvature, P=200 kN compression")
    print("="*80)
    print()

    # Create rectangular section - square 400x400mm
    section = RectangularSection(
        height=400,             # mm
        width=400,              # mm
        cover=40,               # mm
        reinforcementLayers=[
            [52.7, 3, 25],      # [distance from top (mm), num bars, diameter (mm)]
            [347.3, 3, 25]      # Bottom layer
        ],
        transBarDiam=10,        # mm
        spacing=150,            # mm
        numLegsX=2,             # Confinement legs in X
        numLegsY=2              # Shear legs in Y
    )

    # Set material properties
    section.setMaterialProperties(
        fpc=28,                 # MPa
        fy=450,                 # MPa
        fyh=400,                # MPa
        fsu=600,                # MPa
        concreteModel='mc',     # Mander confined
        steelModel='ra'         # Raynor
    )

    # Set member properties
    section.setMemberProperties(
        length=1200,            # mm
        bending='single',
        ductilityMode='uniaxial'  # Rectangular typically uniaxial
    )

    # Set limit states
    section.setLimitStates(
        concreteServiceStrain=0.004,
        steelServiceStrain=0.015,
        concreteDamageStrain=0.018,
        steelDamageStrain=0.060
    )

    # Perform analysis with P-M interaction
    print("Starting analysis...")
    results = section.analyze(axialLoad=200, performInteraction=True)

    # Print key results
    print("\n" + "="*80)
    print("KEY RESULTS - RECTANGULAR SECTION:")
    print("="*80)
    print(f"Nominal Moment:              {results['nominalMoment']:.2f} kN-m")
    print(f"Yield Moment:                {results['yieldMoment']:.2f} kN-m")
    print(f"Ultimate Moment:             {results['momentCurvature']['moment'][-1]:.2f} kN-m")
    print(f"Yield Curvature:             {results['yieldCurvature']:.5f} 1/m")
    print(f"Ultimate Curvature:          {results['momentCurvature']['curvature'][-1]:.5f} 1/m")
    print(f"Section Curvature Ductility: {results['sectionCurvatureDuctility']:.2f}")
    print(f"Yield Displacement:          {results['yieldDisplacement']*1000:.2f} mm")
    print(f"Ultimate Displacement:       {results['ultimateDisplacement']*1000:.2f} mm")
    print(f"Displacement Ductility:      {results['displacementDuctility']:.2f}")
    print(f"Plastic Hinge Length:        {results['plasticHingeLength']:.1f} mm")
    print()
    print("Shear Capacity:")
    print(f"  Concrete contribution:     {results['shearConcrete'][-1]:.2f} kN")
    print(f"  Steel contribution:        {results['shearSteel']:.2f} kN")
    print(f"  Total capacity:            {results['shearCapacity'][-1]:.2f} kN")
    print(f"  Shear failure:             {'YES' if results['shearFailure'] else 'NO'}")
    print()
    print("Buckling:")
    print(f"  Moyer-Kowalsky buckling:   {'YES' if results['bucklingMoyerKowalsky'] else 'NO'}")
    print(f"  Berry-Eberhard buckling:   {'YES' if results['bucklingBerryEberhard'] else 'NO'}")
    print()

    if 'interactionDiagram' in results:
        interaction = results['interactionDiagram']
        print("P-M Interaction:")
        print(f"  Pure tension capacity:     {interaction['pureTension']:.1f} kN")
        print(f"  Pure compression capacity: {interaction['pureCompression']:.1f} kN")
        print(f"  Balanced point:            P={interaction['balancedPoint']['P']:.1f} kN, M={interaction['balancedPoint']['M']:.1f} kN-m")

    print("="*80)

    # Write detailed results
    section.writeResults('CUMBIA_rectangular_results.txt')
    print("\nDetailed results written to 'CUMBIA_rectangular_results.txt'")

    # Plot results
    print("\nGenerating plots...")
    section.plotMaterialModels()
    section.plotResults()

    if 'interactionDiagram' in results:
        section.plotInteractionDiagram()

    print("\n" + "="*80)
    print("RECTANGULAR SECTION TEST COMPLETE!")
    print("="*80)
    print()

    return section


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run complete test suite for CUMBIApy.

    Tests both circular and rectangular sections with full analysis including
    P-M interaction diagrams.
    """
    print("\n")
    print("#"*80)
    print("##" + " "*76 + "##")
    print("##" + " "*20 + "CUMBIApy - COMPLETE TEST SUITE" + " "*26 + "##")
    print("##" + " "*76 + "##")
    print("##" + " "*10 + "Python Implementation of CUMBIA for RC Section Analysis" + " "*14 + "##")
    print("##" + " "*76 + "##")
    print("#"*80)
    print()

    import sys

    # Option to run individual tests
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == 'circular':
            test_circular_section()
        elif test_type == 'rectangular':
            test_rectangular_section()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python CUMBIApy.py [circular|rectangular]")
            print("       (run without arguments to execute all tests)")
    else:
        # Run all tests
        try:
            # Test 1: Circular section
            circular = test_circular_section()

            input("\nPress Enter to continue to rectangular section test...")
            print("\n\n")

            # Test 2: Rectangular section
            rectangular = test_rectangular_section()

            # Summary
            print("\n")
            print("#"*80)
            print("##" + " "*76 + "##")
            print("##" + " "*26 + "TEST SUMMARY" + " "*38 + "##")
            print("##" + " "*76 + "##")
            print("#"*80)
            print()
            print("Both circular and rectangular section analyses completed successfully!")
            print()
            print("Output files generated:")
            print("  - CUMBIA_circular_results.txt")
            print("  - CUMBIA_rectangular_results.txt")
            print()
            print("All plots displayed.")
            print()
            print("#"*80)
            print()
            print("CUMBIApy test suite complete! ✓")
            print()

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user.")
        except Exception as e:
            print(f"\n\nERROR during testing: {e}")
            import traceback
            traceback.print_exc()

