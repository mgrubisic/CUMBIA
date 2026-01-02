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


# Continuing in next message due to length...
