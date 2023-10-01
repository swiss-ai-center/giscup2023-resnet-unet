import yaml

from methods.extract.extract_border import segmentation_model_prediction

EXTRACT_METHODS = {
    "extract_border": segmentation_model_prediction,
}


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    process_params = params["process"]["ptl_unet"]
    extract_params = params["extract"]

    method = extract_params["method"]
    method_kwargs = extract_params[method]
    # Call the selected prepare method function
    if method_kwargs:
        EXTRACT_METHODS[method](
            batch_size=process_params["batch_size"], **method_kwargs
        )
    else:
        EXTRACT_METHODS[method](batch_size=process_params["batch_size"])


if __name__ == "__main__":
    main()
