import java.util.Scanner;

class A_Gravity_Flip{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        sc.nextLine();
        int[] arr = new int[n];

        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
        }

        for (int i = 0; i < n; i++){
            for(int j = 0; j < n - 1; j++){
                if(arr[j] > arr[j + 1]){
                    int dif = arr[j] - arr[j + 1];

                    arr[j + 1] += dif;

                    arr[j] -= dif;
                }
            }
        }

        for(int i : arr) System.out.print(i + " ");
    }
}