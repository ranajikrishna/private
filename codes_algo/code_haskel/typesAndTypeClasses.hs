

-- Types and Type Classes --
-- Page 109


--data Shape = Circle Float Float Float | Rectangle Float Float Float Float deriving (Show)

--Circle : Value Constructor.
--Shape : Type.

--If 'deriving (Show)' is left out then > :Circle 10 10 2 would give error because Shape is not an instance of the Type Class Show. However, > : area(Circle 10 10 2) would still give us a value since Float is an instance of the type class Show. 

--area :: Shape -> Float
--area (Circle _ _ r) = pi * r ^ 2

data Point = Point Float Float deriving (Show)  --Point is a Type; it is a type instance derived from the type class Show.

data Shape = Circle Point Float | Rectangle Point Point deriving (Show)  -- Shape is a Type; its a type instance derived from the type class Show.

area :: Shape -> Float          -- Type Signature, area is a fxn. and Shape is a Type (*not* a typr varoable beacuse it *cannot* take in any type).  

area (Circle _ r) = pi * r ^ 2	
area (Rectangle (Point x1 y1) (Point x2 y2)) = (x2 - x1) * (y2 - y1)
--Try > :area (Circle (Point 10 10) 2)

--nudge :: Shape -> Float -> Float -> Shape
nudge :: Shape -> Point -> Shape
--nudge (Circle (Point x1 y1) r) x2 y2  = Circle (Point (x1+x2) (y1+y2)) r
nudge (Circle (Point x1 y1) r) (Point x2 y2)  = Circle (Point (x1 + x2) (y1 + y2)) r
nudge (Rectangle (Point x1 y1) (Point x2 y2)) (Point a b) = Rectangle (Point (x1+a) (y1+b)) (Point (x2+a) (y2+b))
-- > nudge (Circle (Point 10 10) 2) (Point 1 1)

baseCircle :: Float -> Shape
baseCircle r = Circle ( Point 0 0) r

baseRectangle :: Float -> Float -> Shape
baseRectangle width height = Rectangle (Point 0 0) (Point width height)

-- try :> nudge (baseRectangle 40 100) (Point 60 23)

-- Record Syntax --

--data Person = Person String String Int Float String String deriving (Show)

--firstName :: Person -> String
--firstName (Person a _ _ _ _ _ ) = a

--age :: Person -> Int
--age (Person _ _ a _ _ _) = a

--let guy = Person "Ranaji" "Krishna" 31 175 "26Apr1983" "Brown"
-- >firstName guy
--Even if we comment deriving (Show) out, the above two will print an output since both Int and String are instances derived from the type class show.

data Person1 = Chap String String Int Float String String deriving (Show)

weight :: Person1 -> Float
weight (Chap _ _ _ a _ _) = a
 
--let guy = Chap "Ranaji" "Krishna" 31 175 "26Apr1983" "Brown"
-- >weight guy
--NOTE: data Person = Chap String String ... etc will NOT work since the type Person has already been defined to have value constructor Person.
--IMP: If you change the Float to Int in the type declaration, and leave the type signature as is, then we get an error in compilation due to mismatch between type "Float and actual type Int".


--We can condense the above to the followig:
data Person = Person { firstName  :: String
		     , secondName :: String
		     , age        :: Int
		     , mass       :: Float
		     , birthDate  :: String
		     , colour     :: String
		     } deriving (Show)

--let guy = Person "Ranaji" "Krishna" 31 175 "26Apr1983" "Brown"
-- > colour guy will print "Brown"
--Even if we comment deriving (Show) out, the above two will print an output since the String is an instance is derived from the type class show.

-- Type Parameters --

data Maybe a = Nothing | Just a

--data Car = Car { model :: String
--       	 , trim  :: String
--	         , year  :: Int
--	         } deriving (Show)

--tellCarModel :: Car -> String
--tellCarModel (Car {model= a , trim = b, year = c }) = a
--tellCarModel (Car a _ _) = a
-- Both of the above two work.
--Even if we comment deriving (Show) out, the above two will print an output since the String is an instance is derived from the type class show.

data Car a b c = Mobile { model :: a		-- Car   : Type Constructor
		        , trim  :: b		-- mobile: Value Constructor	
		        , year  :: c		-- a,b,c : Type Parameter (or Type Variable because it can take in any Type).
		        } deriving (Show) 	-- model, trim, year : Fxns (Polymorphic fxns. because they can use type variables).

-- let vehicle = Mobile "Ford" "Fusion" 2013
-- >model vehicle will give "Ford"
-- let vehicle = Mobile 2090 "Fusion" "Jesus"
-- >model vehicle will give 2090

tellCar :: (Show a) => Car Integer a b  -> a   
--By doing this we can use any type, in the position of a, which is an instance of Show class. For example we could use Integer, String, etc. In other words we are saying that a can be of any type of instance of class show.
--NOTE: So apart from desribing fxns. model, trim and year during declaratoin, we can declare other fxns. like tellCar as well. 
--NOTE: Show is a *Type Class*, and a is a type that is an instance of a type class. Everything before the symbol => is called a *class constaint*. The tpe signature (==) :: (Eq a) => a -> a -> bool is **READ AS**, == is a fxn. of two values of the same type and returns a bool. The two values are an instance of the type class Eq. 

tellCar (Mobile _ a _) = a
-- > let vehicle Mobile 2013 "Fusion" "Jesus" and let vehicle Mobile 2098 3456 0982 will also work.
--Note: This will also work, tellCar::(Show a) => Car Integer a b -> Integer, when the fxn., tellCar (Mobile a _ _) = a is defined.

-- Vector Von Doom --

data Vector a = Vector a a a deriving (Show)

--For Vector addition--
vplus :: (Num a) => Vector a -> Vector a -> Vector a
--(Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)
vplus (Vector i j k) (Vector l m n) = Vector (i+l) (j+m) (k+n)

dotProd :: (Num a) => Vector a -> Vector a -> a
dotProd (Vector i j k) (Vector l m n) = (i * l) + (j * m) + (k * n)
-- (Vector i j k ) `dotProd` (Vector l m n) =  Vector (i * l) + Vector (j * m) + Vector (k * n)
-- We cannot use, for example dotProd :: (Num a) => Vector a -> Vector a -> Num, as type signature. This is because Num is **not** a Type, it is a Type Class. We can use dotProd::(Num a) => Vector a -> Vector a -> nteger with the fxn dotProd (Vector i j k) (Vector  m n) = 8.

vMult :: (Num r) => Vector r -> r -> Vector r
vMult (Vector i j k) m = Vector (i*m) (j*m) (k*m)

-- Equating People --
data Person2 = Fellow { bgnName  :: String
		      , endName  :: String
		      , old      :: Int
		      } deriving (Eq)

mikeD =  Fellow {bgnName = "Michael", endName = "Jackson", old = 43}
adRock = Fellow {bgnName = "Adam", endName = "Christ", old = 41}
mca =    Fellow {bgnName = "Adam", endName = "Fine", old = 44}

--mikeD ==  Fellow {bgnName = "Michael", endName = "Jackson", old = 43} would give True
--let guys = [mikeD, adRock, mca], elem mikeD guys would give True.
--Since Fellow is not an instance derived from the Show classs, we will get an error message when we type, for example mikeD in the console. However if we add the Show type class, we can get an output.

-- Show me how to Read --
data Person3 = Human  { goodName :: String
		      , surName  :: String
		      , life     :: Int
		      } deriving (Eq,Show,Read)

mikeJack =  Human {goodName = "Michael", surName = "Jackson", life = 43}

--read "Human {goodName = \"Ranaji\", surName = \"Krishna\", life = 31}" :: Person3
--read converts strings to values **of out type**, hence :: Person3 is **not in** "".

--This will work 
mysteryDude = "Human { goodName = \"Ranaji\" " ++
	      	       ", surName  = \"Krishna\"" ++
                     ", life = 31}"
--when we do
--read mysteryDude :: Person3 in the gchi console.
--mysteryDude == MikeD will give False

-- Order in the Court --

data Boolu = Falseu | Trueu deriving (Ord, Eq)

data Day = Monday | Tuesday | Wednesday | Thursday | Friday deriving (Ord, Show, Eq)

--Type Synonyms
type Stringi = [Char]
--This is **not** a type creation (that is done with the keyword 'data'). This rather defines a synonym for an existig type.

toUpperString :: Stringi -> Stringi
toUpperString a = a ++ " is the best!" 


--Making our Phone Book Prettier
phoneBook :: [(String, String)]

phoneBook = [("Lahiru", "07898761893")
	    ,("Imran", "07886716103")
	    ,("Shyam", "07816462102")
	    ,("Aashish", "07545270104")
	    ]

type Name = String
type PNum = String

phBook :: [(Name,PNum)]
phBook =    [("Lahiru", "07898761893")
	    ,("Imran", "07886716103")
	    ,("Shyam", "07816462102")
	    ,("Aashish", "07545270104")
	    ]

type PBookType = [(Name,PNum)]
pBook :: PBookType
pBook =     [("Lahiru", "07898761893")
	    ,("Imran", "07886716103")
	    ,("Shyam", "07816462102")
	    ,("Aashish", "07545270104")
	    ]

inPhoneBook :: Name -> PNum -> PBookType -> Bool
inPhoneBook nameVar pNumVar phoneBookVar = (nameVar, pNumVar) `elem` phoneBookVar 
--Both inPhoneBook "Lahiru" "07898761893" pBook and inPhoneBook "Lahiru" "07898761893" phBook work.

--Parameterizing Type Synonyms --

--type K = Int
--type V = String
type AssocList K V = [(K,V)]

addLocus :: (Eq k) => k -> AssocList k v -> AssocList k v
addLocus k AssocList l m = AssocList (k + l) m





























