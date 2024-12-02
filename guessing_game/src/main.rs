use std::io;
use rand;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    loop {
        let secret_number = rand::random::<u32>(); // rand::thread_rng().gen_range(1, 100);

        println!("Please input your guess.");

        let mut guess = String::new();
    
        io::stdin()
        .read_line(&mut guess)
        .expect("Failed to read line");
    
        let guess: u32 = guess.trim().parse().expect("Please type a number!");
    
        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => println!("You win!"),
        }
    
        println!("You guessed: {}", guess);
        println!("The secret number is: {secret_number}");    
    }
}