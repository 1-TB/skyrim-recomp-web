// The app's entry point: everything runs in the Rust static library (hosts/host-ios), which
// hands the main thread to UIKit and never returns.
extern void skyrt_ios_main(void);

int main(void) {
    skyrt_ios_main();
    return 1;
}
