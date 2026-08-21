from setuptools import find_namespace_packages, setup

setup(
    name="hsem",
    version="7.1.7",
    description="Personal Home Assistant energy planner for Huawei and PowMr storage",
    packages=find_namespace_packages(include=["custom_components.*"]),
    include_package_data=True,
    zip_safe=False,
)
