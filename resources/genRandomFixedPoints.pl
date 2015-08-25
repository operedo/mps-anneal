my $i=0;
my @randomnum;
my @randomnum2;
my @randomnum3;
while($i<$ARGV[0]){
	my $rn1=int(rand(99));
	my $rn2=int(rand(99));
	my $rn3=int(rand(99));
	print $rn1," ",$rn2," ",$rn3,"\n";
	$i++;	
}
