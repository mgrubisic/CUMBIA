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


# Continue with more methods in next edit...
