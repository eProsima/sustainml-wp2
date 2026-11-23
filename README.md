# SustainML – Work Package 2 (WP2)

[![SustainML](.figures/SustainML_GitHub.png)](https://sustainml.eu/)

<br>

<div class="menu" align="center">
    <strong>
        <a href="https://eprosima.com/index.php/downloads-all">Download</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://sustainml.readthedocs.io/en/latest">Docs</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://sustainml.eu/showroom/news">News</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://www.linkedin.com/company/eprosima/">LinkedIn</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="mailto:info@eprosima.com">Contact Us</a>
    </strong>
</div>

<br><br>

<div class="badges" align="center">
    <a href="https://github.com/eProsima/sustainml-wp2/releases"><img alt="Releases" src="https://img.shields.io/github/v/release/eProsima/sustainml-wp2?sort=semver"/></a>
    <a href="https://github.com/eProsima/sustainml-wp2/issues"><img alt="Issues" src="https://img.shields.io/github/issues/eProsima/sustainml-wp2.svg"/></a>
    <a href="https://github.com/eProsima/sustainml-wp2/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/eProsima/sustainml-wp2.svg"/></a>
    <a href="https://github.com/eProsima/sustainml-wp2/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/eProsima/sustainml-wp2.svg"/></a>
    <a href="https://SustainML.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://img.shields.io/readthedocs/sustainml/latest"/></a>
</div>

<br><br>

This repository contains components developed as part of **Work Package 2 (WP2)** of the SustainML project.

WP2 focuses on **evaluating the proposed machine learning model on the hardware** chosen by the user.

## Scope

This repository contains two nodes of the SustainML Framework:

- **HW Constraints** (`hw_constraints_node.py`): collects the hardware chosen by the user and the maximum memory
  footprint of the problem.
- **HW Resources Provider** (`hw_resources_provider_node.py`): computes the latency, power and memory footprint of the
  model on that hardware, in one of three ways:
  - Transformers models on the processing-in-memory and GPU architectures of the UPMEM LLM simulator (`upmem_llm_framework`).
  - YOLO models on the Axelera Edge 130p board, from the measurements of each model on the board (`hw_provider_axelera`).
  - U-Net models on the FPGA `xczu19eg-ffvb1517-2-i`: the latency and energy predicted by DFKI's predictor, for the
    217 U-Nets of the catalogue as for any other U-Net (`hw_provider_fpga`).

The memory footprint of the models (weights plus working memory) is computed in `model_memory.py`.

## Usage

The WP2 nodes are not intended to operate independently: they are integrated into the SustainML pipeline as part of
the `sustainml_modules` package of the [SustainML Library](https://github.com/eProsima/SustainML-Library), where this
repository is a Git submodule.

For installation, configuration, and execution instructions, refer to the main framework repository:

https://github.com/eProsima/SustainML-Framework

## Third-party code

The `upmem_llm_framework` directory contains a modified copy of the UPMEM LLM Framework, taken from eProsima's fork https://github.com/eProsima/upmem_llm_framework (commit `f4ed113`).
It is distributed under the MIT License (`upmem_llm_framework/LICENSE`), Copyright (c) 2024 UPMEM.
See `upmem_llm_framework/README.md` for the list of local modifications.

The `hw_provider_fpga/vendor/sustain_ml_predictor` directory contains DFKI's U-Net latency and energy predictor, its
predictor models, and the U-Net models and measurements it was trained on, developed within the SustainML project.
