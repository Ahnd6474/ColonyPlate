import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="YOLOv11 detect weights (best.pt)")
    ap.add_argument("--sessions", type=str, default="./sessions", help="folder to store sessions + label_library.json")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    from colony_tool.app import build_app

    demo = build_app(args.weights, args.sessions)
    demo.launch(server_name=args.host, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
